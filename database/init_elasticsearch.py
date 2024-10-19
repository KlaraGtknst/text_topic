from elasticsearch import ApiError, ConflictError, Elasticsearch
#from text_embeddings.preprocessing.read_pdf import *
from elasticsearch.helpers import bulk
from sentence_transformers import SentenceTransformer
from data.files import pdf_to_str, get_hash_file, extract_text_from_pdf
from constants import *
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
    - path: the path to the document on the local maschine.

    cf. https://www.elastic.co/guide/en/elasticsearch/reference/current/dense-vector.html for information about dense vectors and similarity measurement types
    '''
    client.indices.create(index=DB_NAME, mappings={
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
            "path": {
                "type": "keyword",
            },
            "file_name": {
                 "type": "text",
            },
        },
    })


def initialize_db(src_path, client_addr=CLIENT_ADDR):
    print('-' * 80)

    # Create the client instance
    client = Elasticsearch(client_addr)
    print('finished creating client')

    # delete old index and create new one
    client.options(ignore_status=[400, 404]).indices.delete(index=DB_NAME)
    init_db(client)
    print('finished deleting old and creating new index')

    # insert embeddings
    insert_embeddings(src_path, client)
    print('finished inserting embeddings')

    return client


def scanRecurse(baseDir: str):
    baseDir = baseDir.split('*')[0] if '*' in baseDir else baseDir

    for entry in os.scandir(baseDir):
        if entry.is_file():
            yield os.path.join(baseDir, entry.name)
        else:   # recurse needs from, otherwise generator object is returned
            yield from scanRecurse(entry.path + '/')


def insert_embeddings(src_path: str, client: Elasticsearch):
    print('started with generate_models_embedding()')
    model = SentenceTransformer('sentence-transformers/msmarco-MiniLM-L-12-v3')

    for path in scanRecurse(baseDir=src_path):
        if not (path.endswith('.pdf') or path.endswith('.txt')):
            continue

        text = extract_text_from_pdf(path) if path.endswith('.pdf') else open(path, 'r').read()
        id = get_hash_file(path)

        try:
            embedding = model.encode(text[0])
            #print('embedding: ', os.path.basename(path))#embedding)
            client.update(index=DB_NAME, id=id, body={'doc': {'embedding': embedding, 'text': text[0],
                                                              'path': path, 'file_name': os.path.basename(path)},
                                                      'doc_as_upsert': True})
        except Exception as e:
            print('error in embedding: ', path, e)
            continue


def main(src_path: str, client_addr=CLIENT_ADDR):
    initialize_db(src_path, client_addr=client_addr)


if __name__ == '__main__':
    #args = arguments()
    src_path = TEST_TRAINING_PATH#args.directory

    client = initialize_db(src_path, client_addr=CLIENT_ADDR)
    res = client.search(index=DB_NAME, body={
        'size': 10,
        'query': {
            'match_all': {}
        }
    })
    print('result: ', res)