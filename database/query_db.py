from collections import defaultdict

import matplotlib.pyplot as plt
from wordcloud import WordCloud

import constants
from constants import *
from utils.os_manipulation import save_or_not
from visualization.two_d_display import scatter_documents_2d


def search_db(client, num_res: int = 10):
    """
    Returns ten first documents in the database.
    :param client: Elasticsearch client
    :param num_res: Number of results to return
    :return: result of the query
    """
    res = client.search(index=DatabaseAddr.DB_NAME, body={
        'size': num_res,
        'query': {
            'match_all': {}
        }
    })
    return res


def get_num_indexed_documents(client):
    """
    Returns the number of documents in the database.
    :param client: Elasticsearch client
    :return: number of documents
    """
    client.indices.refresh(index=DatabaseAddr.DB_NAME)
    count = int(client.cat.count(index=DatabaseAddr.DB_NAME, format="json")[0]["count"])
    return count


def obtain_directories(client):
    """
    Returns a set of all directories in the database.
    :param client: Elasticsearch client
    :return: list of directories
    """
    count = get_num_indexed_documents(client)
    res = search_db(client, num_res=count)
    return set([r['_source']['directory'] for r in res['hits']['hits']])


def get_directory_content(client, directory: str):
    """
    Returns a list of all texts in a given directory (only this directory and not its children).
    :param client: Elasticsearch client
    :param directory: Directory to get content of
    :return: None
    """
    res = client.search(index=DatabaseAddr.DB_NAME, body={
        '_source': ['text'],
        'size': get_num_indexed_documents(client),
        'query': {
            'match': {
                'directory': directory
            }
        }
    })
    texts = [r['_source']['text'] for r in res['hits']['hits']]
    return texts


def display_directory_content(client, directory: str, save_path: str = None):
    """
    Displays a wordcloud of the content of a given directory.
    If save_path is not None, saves the wordcloud as a .png file.
    :param client: Elasticsearch client
    :param directory: Directory to display content of
    :param save_path: Path to save the wordcloud
    :return: None
    """
    sentences = get_directory_content(client, directory)
    text = ' '.join(sentences)
    wordcloud = WordCloud(max_font_size=40).generate(text)
    plt.figure()
    plt.title('Wordcloud of directory: ' + directory)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    save_or_not(plt, file_name='wordcloud_' + directory + '.svg', save_path=save_path, format='svg')
    plt.show()


def scatter_dir_content(client, save_path: str = None):
    """
    Displays a 2D scatter plot of the documents in the database.
    The documents are represented by their embeddings.
    Each document is colored according to its parent directory.
    The plot is saved as a .png file if save_path is not None.
    :param client: Elasticsearch client
    :param save_path: Path to save the scatter plot
    :return: None
    """
    scatter_documents_2d(client, save_path=save_path)


# should work, since used in NER/clustering_NE.py
def get_named_entities_for_docs(client, key_name: str, nested_field_path: str = "named_entities",
                                es_request_limit: int = 10000):
    """
    Fetch named entities of the specified category using the scroll API for large datasets.
    :param client: Elasticsearch client
    :param nested_field_path: Path to the nested field (e.g., "parent.child").
    :param key_name: The specific key to retrieve values for.
    :param es_request_limit: Number of documents to fetch in each Elasticsearch request at a time.
    :return: result of the query, a map of named entities to documents
    """
    named_entities = []
    doc_map = defaultdict(list)  # Map named entities to documents

    query = {
        "size": es_request_limit,
        "_source": [f"{nested_field_path}.{key_name}"],
        "query": {
            "nested": {
                "path": nested_field_path,
                "query": {
                    "exists": {
                        # Check if the 'value' field exists in the nested field
                        "field": f"{nested_field_path}.{key_name}"
                    }
                },
            }
        }
    }

    response = client.search(index=constants.DatabaseAddr.DB_NAME, body=query, scroll="2m")
    scroll_id = response["_scroll_id"]

    # Process the first batch of results
    while True:
        hits = response["hits"]["hits"]
        if not hits:
            break

        for doc in hits:
            doc_id = doc["_id"]
            entities = doc["_source"].get("named_entities", {}).get(key_name, [])
            named_entities.extend(entities)
            for entity in entities:
                doc_map[entity].append(doc_id)

        # Fetch the next batch of results
        response = client.scroll(scroll_id=scroll_id, scroll="2m")

    # Clear the scroll context to free resources on the server
    client.clear_scroll(scroll_id=scroll_id)
    return named_entities, doc_map


def get_texts_from_docs(client, es_request_limit: int = 10000):
    """
    Fetch named entities of the specified category using the scroll API for large datasets.
    :param client: Elasticsearch client
    :param es_request_limit: Number of documents to fetch in each Elasticsearch request at a time.
    :return: result of the query, a map of named entities to documents
    """

    texts = []

    query = {
        "size": es_request_limit,
        '_source': ['text'],
        'query': {
            'match_all': {}
        }
    }

    response = client.search(index=constants.DatabaseAddr.DB_NAME, body=query, scroll="2m")
    scroll_id = response["_scroll_id"]

    # Process the first batch of results
    while True:
        hits = response["hits"]["hits"]
        if not hits:
            break

        for doc in hits:
            texts.append(doc['_source']['text'])

        # Fetch the next batch of results
        response = client.scroll(scroll_id=scroll_id, scroll="2m")

    # Clear the scroll context to free resources on the server
    client.clear_scroll(scroll_id=scroll_id)
    return texts


# if __name__ == '__main__':
#     es_db = db.Database()
#
#     # get client of existing database
#     client = es_db.get_client()
#
#     # obtain directories & display content
#     display_directory_content(client, directory='Weapons', save_path=Paths.SERVER_PLOTS_SAVE_PATH)
#
#     # scatter plot of documents highlighting directories
#     scatter_dir_content(client, save_path=Paths.SERVER_PLOTS_SAVE_PATH)
#
#     # get named entities for documents
#     nested_field_path = "named_entities"
#     key_name = "GPE"  #"ORG"
#     named_entities = get_named_entities_for_docs(client, nested_field_path, key_name)
#     print(f"Named entities for key '{key_name}': {named_entities}")
#
#     print('finished')
