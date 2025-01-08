import spacy
from elasticsearch import ApiError, ConflictError, Elasticsearch, NotFoundError
from elasticsearch.helpers import bulk
from sentence_transformers import SentenceTransformer
from data.files import pdf_to_str, get_hash_file, extract_text_from_pdf
from constants import *
from utils.os_manipulation import scanRecurse
from NER import named_entity_recognition
import os

'''------initiate, fill and search in database-------
run this code by typing and altering the path:
    python3 init_elasticsearch.py
'''


def init_db(client: Elasticsearch):
    '''
    :param client: Elasticsearch client
    :return: None

    This function initializes the database by creating an index (i.e. the structure for an entry of type DB_NAME database).
    The index contains the following fields:
    - text: the text of the document. The text is not tokenized, stemmed etc.
    - path: the path to the document on the local machine.
    - embedding: the SentenceTransformer embedding of the text.
    - directory: the parent directory of the document.
    - file_name: the name of the document.

    cf. https://www.elastic.co/guide/en/elasticsearch/reference/current/dense-vector.html for information about dense vectors and similarity measurement types
    '''
    client.indices.create(index=DB_NAME, body={
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
                "named_entities": {
                    "type": "nested",
                },
            },
        }
    })


def initialize_db(client_addr=CLIENT_ADDR, src_path="", create_db=True):
    """
    Initialize the database by creating an index and inserting the embeddings of the documents in the database.
    :param src_path: Path to the directory containing the documents (.txt and .pdf)
    :param client_addr: Address of the Elasticsearch client
    :param init_db: Boolean indicating whether to create the database (true) or just return existing client (false)
    :return: Elasticsearch client
    """
    print('-' * 80)

    # Create the client instance
    client = Elasticsearch(client_addr)
    print('finished creating client')

    if create_db:
        # delete old index and create new one
        # client.options(ignore_status=[400, 404]).indices.delete(index=DB_NAME)
        # init_db(client)
        # print('finished deleting old and creating new index')

        if src_path != "":
            try:
                # insert embeddings
                insert_embeddings(src_path, client)
                print('finished inserting embeddings')
            except Exception as e:
                print(e)
        else:
            raise ValueError('no path given')

    return client


def insert_embeddings(src_path: str, client: Elasticsearch):
    """
    Insert SentenceTransformer (SBERT) embeddings of documents (.txt and .pdf) in the database.
    https://www.sbert.net/

    :param src_path: Path to the directory containing the documents (.txt and .pdf)
    :param client: Elasticsearch client
    :return: -
    """
    print('started with insert_embeddings()')
    model = SentenceTransformer('sentence-transformers/msmarco-MiniLM-L-12-v3')
    ner = named_entity_recognition.NamedEntityRecognition()

    for path in scanRecurse(baseDir=src_path):


        if path.endswith('.png') or path.endswith('.jpg') or path.endswith('.jpeg'):
            text = path.split('/')[-1].split('.')[0]
        elif path.endswith('.pdf') or path.endswith('.txt'):
            text, success = extract_text_from_pdf(path) if path.endswith('.pdf') else open(path, 'r').read()
            if not success:
                text = 'Error extracting text from pdf.'
        else:
            print('current path did not work:', path)
            continue

        text = ' '.join(text)
        id = get_hash_file(path)
        named_entities = ner.get_named_entities_dictionary(text=text)

        doc = {'embedding': model.encode(text), 'text': text, 'path': path, 'file_name': os.path.basename(path),
               'directory': os.path.dirname(path).split('/')[-1], 'named_entities': named_entities}

        try:
            # insert document in database if it does not exist, else update it
            client.update(index=DB_NAME, id=id, doc=doc, doc_as_upsert=True)

        except Exception as e:
            print('error in embedding: ', e)
            continue


def main(src_path: str, client_addr=CLIENT_ADDR):
    initialize_db(src_path=src_path, client_addr=client_addr, create_db=False)


# if __name__ == '__main__':
#     #args = arguments()
#     src_path = SERVER_PATH  #TEST_TRAINING_PATH#args.directory
#
#     client = initialize_db(client_addr=CLIENT_ADDR, create_db=True, src_path=src_path)
#     res = client.search(index=DB_NAME, body={
#         'size': 10,
#         'query': {
#             'match_all': {}
#         }
#     })
#     print('result: ', res)
#     print('text: ', res['hits']['hits'][0]['_source']['text'])
#     print('finished')
