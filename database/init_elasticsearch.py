import logging
import os
from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
from sentence_transformers import SentenceTransformer
from NER import named_entity_recognition
from constants import *
from data.caption_images import ImageCaptioner
from data.files import get_hash_file, extract_text_from_pdf, extract_text_from_txt
from utils.logging_utils import init_debug_config
from utils.os_manipulation import scan_recurse

'''------initiate, fill and search in database-------
run this code by typing and altering the path:
    python3 init_elasticsearch.py
'''

logger = logging.getLogger(__name__)


class ESDatabase:
    def __init__(self, client_addr: str = DatabaseAddr.CLIENT_ADDR.value):
        self.client = Elasticsearch(client_addr, request_timeout=100)
        init_debug_config(log_filename='init_elasticsearch_', on_server=True)

    def get_es_client(self):
        return self.client

    def init_db(self):
        """
        This function initializes the database by creating an index (i.e. the structure for an entry of type DB_NAME database).
        The index contains the following fields:
        - text: the text of the document. The text is not tokenized, stemmed etc.
        - path: the path to the document on the local machine.
        - embedding: the SentenceTransformer embedding of the text.
        - directory: the parent directory of the document.
        - file_name: the name of the document.

        cf. https://www.elastic.co/guide/en/elasticsearch/reference/current/dense-vector.html for information about dense vectors and similarity measurement types
        """
        logger.info('Started creating index')

        self.client.indices.create(index=DatabaseAddr.DB_NAME.value, body={
            "mappings": {
                "properties": {
                    "embedding": {
                        "type": "dense_vector",
                        "dims": 384,
                        "index": True,
                        "similarity": "cosine",
                    },
                    "text": {
                        "type": "text",
                    },
                    "directory": {
                        "type": "text",
                    },
                    "path": {
                        "type": "keyword",
                    },
                    "file_name": {
                        "type": "text",
                    },
                    "file_type": {
                        "type": "text",
                    },
                    "named_entities": {
                        "type": "nested",
                    },
                },
            }
        })
        logger.info('Finished creating index')

    def initialize_db(self, src_path="", delete_old_index=False):
        """
        Initialize the database by creating an index and inserting the embeddings of the documents in the database.
        Only call this function if you want to create a NEW database.
        Use `client = Elasticsearch(client_addr)` to connect to an existing database.
        :param src_path: Path to the directory containing the documents (.txt and .pdf)
        :param delete_old_index: If True, the old index is deleted and a new one is created
        :return: Elasticsearch client
        """
        logger.info('started with initialize_db()')

        # delete old index and create new one
        if delete_old_index:
            self.client.options(ignore_status=[400, 404]).indices.delete(index=DatabaseAddr.DB_NAME.value)
            self.init_db()
            logger.info('deleted old index and created new one')

        if src_path != "":
            try:
                # insert embeddings
                self.insert_metadata(src_path)

            except Exception as e:
                logger.error(f'error in inserting metadata: {e}')
                raise e
        else:
            raise ValueError('no path given')

        return self.client

    def insert_text_related_fields(self, src_path: str):
        """
        Insert captions of images and texts of documents (.txt and .pdf) in the database.
        Since text is used for the embeddings and named entities, these are also updated in the database.
        The embeddings are generated using a SentenceTransformer (SBERT).
        For more information: https://www.sbert.net/ (21.01.2025)

        :param src_path: Path to the directory containing the documents (.txt and .pdf)
        :return: -
        """
        # Create the client instance
        logger.info('start with insert_text_related_fields()')
        image_captioner = ImageCaptioner()
        ner = named_entity_recognition.NamedEntityRecognition()
        model = SentenceTransformer('sentence-transformers/msmarco-MiniLM-L-12-v3')

        for path in scan_recurse(base_directory=src_path):
            text = self.obtain_text_from_file(image_captioner, path)

            id = get_hash_file(path)
            limit = min(10 ** 6, len(text))  # nlp.max_length: https://spacy.io/api/language
            named_entities = ner.get_named_entities_dictionary(text=text[:limit])

            update_doc = {'text': text, 'named_entities': named_entities, 'embedding': model.encode(text)}

            try:
                # insert document in database if it does not exist, else update it
                self.client.update(index=DatabaseAddr.DB_NAME.value, id=id, doc=update_doc, doc_as_upsert=True)

            except Exception as e:
                logging.error('error in embedding: ', e)
                continue

    def insert_text_related_fields_bulk(self, src_path: str):
        """
        Insert captions of images and texts of documents (.txt and .pdf) in the database.
        Since text is used for the embeddings and named entities, these are also updated in the database.
        The embeddings are generated using a SentenceTransformer (SBERT).
        For more information: https://www.sbert.net/ (21.01.2025)

        :param src_path: Path to the directory containing the documents (.txt and .pdf)
        :return: -
        """
        logging.info('start with insert_text_related_fields_bulk()')

        image_captioner = ImageCaptioner()
        ner = named_entity_recognition.NamedEntityRecognition()
        logging.info('created image_captioner and ner instance')
        model = SentenceTransformer('sentence-transformers/msmarco-MiniLM-L-12-v3')
        logging.info('created embedding model instance')

        paths = scan_recurse(base_directory=src_path)
        texts = [self.obtain_text_from_file(image_captioner, path) for path in paths]
        logging.info('obtained texts')
        named_entities_bulk = ner.get_named_entities_batch(texts)
        logging.info('obtained named entities bulk')

        actions = []  # List to hold bulk actions
        logging.info('created empty actions list')

        for idx, named_entities in enumerate(named_entities_bulk):
            id = get_hash_file(paths[idx])  # Assuming paths is the list of file paths
            update_doc = {
                '_op_type': 'update',
                '_index': DatabaseAddr.DB_NAME.value,
                '_id': id,
                'doc': {
                    'text': texts[idx],
                    'named_entities': named_entities,
                    'embedding': model.encode(texts[idx]),
                },
                'doc_as_upsert': True,
            }
            actions.append(update_doc)
        logging.info('finished creating actions list')
        # Execute bulk update
        try:
            success, failed = bulk(self.client, actions, chunk_size=500)
            logger.info(f"Successfully executed {success} actions.")
            if failed:
                logger.warning(f"Failed actions: {failed}")
        except Exception as e:
            logger.error(f"Bulk operation failed: {e}")

    def obtain_text_from_file(self, image_captioner, path: str):
        """
        Function to obtain the text from a file.
        :param image_captioner: Instance of the ImageCaptioner class
        :param path: Path to the file
        :return: text (string)
        """
        try:
            if path.endswith('.pdf'):
                text, success = extract_text_from_pdf(path, find_caption=True)
            elif path.endswith('.txt'):
                text, success = extract_text_from_txt(path)
            elif path.endswith('.png') or path.endswith('.jpg') or path.endswith('.jpeg'):
                text = image_captioner.caption_image(path)  # generate caption for image
            else:  # any other file type
                text = path.split('/')[-1].split('.')[0]
            return text
        except Exception as e:
            return str(e)

    def insert_metadata(self, src_path: str):
        """
        Function to insert metadata of documents in the database.
        This metadata includes the path, file name, directory, file type, and parent directory.

        :param src_path: Path to the directory containing the documents (.txt and .pdf)
        :return: -
        """
        logger.info('started with insert_metadata()')

        for path in scan_recurse(base_directory=src_path):

            id = get_hash_file(path)
            doc = {'path': path, 'file_name': os.path.basename(path), 'directory': os.path.dirname(path).split('/')[-1],
                   'file_type': path.split('.')[-1]}
            try:
                # insert document in database if it does not exist, else update it
                self.client.update(index=DatabaseAddr.DB_NAME.value, id=id, doc=doc, doc_as_upsert=True)

            except Exception as e:
                logger.error(f'error in updating document {path}. Error is: {e}')
                continue

        logger.info('finished inserting metadata')
