from collections import defaultdict
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import constants
from constants import *
from utils.os_manipulation import save_or_not, exists_or_create
from visualization.two_d_display import scatter_documents_2d
from elasticsearch import Elasticsearch


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


def get_directory_content(client, index: str, directory: str, scroll: str = "2m", batch_size: int = 1000):
    """
    Returns a list of all texts in a given directory (only this directory and not its children).
    :param client: Elasticsearch client
    :param index: Name of the Elasticsearch index
    :param directory: Directory to get content from
    :param scroll: Scroll duration (default: 2 minutes)
    :param batch_size: Number of documents fetched per scroll request
    :return: List of all texts in the directory
    """
    # Initialize the query to fetch documents in the specified directory
    query = {
        "_source": ["text"],  # Only fetch the "text" field
        "size": batch_size,
        'query': {
            'match': {
                'directory': directory
            }
        }
    }

    # Perform the initial scroll search
    response = client.search(index=index, body=query, scroll=scroll)
    scroll_id = response["_scroll_id"]
    texts = []

    # Continue fetching results until no more documents are returned
    while True:
        hits = response["hits"]["hits"]
        if not hits:
            break

        texts.extend([hit["_source"]["text"] for hit in hits if "text" in hit["_source"] and hit["_source"]["text"]])

        # Fetch the next batch of results
        response = client.scroll(scroll_id=scroll_id, scroll="2m")

    # Clear the scroll context to free up resources
    client.clear_scroll(scroll_id=scroll_id)

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
    sentences = get_directory_content(client=client, directory=directory, index=constants.DatabaseAddr.DB_NAME.value)
    text = ' '.join(sentences)
    wordcloud = WordCloud(max_font_size=40).generate(text)
    plt.figure()
    plt.title('Wordcloud of directory: ' + directory)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    exists_or_create(path=save_path)
    save_or_not(plt, file_name='wordcloud_' + directory + '.svg', save_path=save_path, format='svg')
    #plt.show()
    plt.close()


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

    response = client.search(index=constants.DatabaseAddr.DB_NAME.value, body=query, scroll="2m")
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

    response = client.search(index=constants.DatabaseAddr.DB_NAME.value, body=query, scroll="2m")
    scroll_id = response["_scroll_id"]

    # Process the first batch of results
    while True:
        hits = response["hits"]["hits"]
        if not hits:
            break

        for doc in hits:
            if 'text' in list(doc['_source'].keys()):
                texts.append(doc['_source']['text'])

        # Fetch the next batch of results
        response = client.scroll(scroll_id=scroll_id, scroll="2m")

    # Clear the scroll context to free resources on the server
    client.clear_scroll(scroll_id=scroll_id)
    return texts


def get_column_values_scroll(client, index: str, column: str, scroll_time: str = "2m", batch_size: int = 1000):
    """
    Retrieves all values from a specific column in an Elasticsearch index using a scroll query.

    :param client: Elasticsearch client
    :param index: Name of the Elasticsearch index
    :param column: Column (field) to fetch values from
    :param scroll_time: Time to keep the scroll context alive (default: 2 minutes)
    :param batch_size: Number of documents to retrieve per batch (default: 1000)
    :return: List of all values from the specified column
    """
    # Initialize an empty list to store values
    values = []

    # Initial search to start the scroll
    response = client.search(
        index=index,
        body={
            "size": batch_size,
            "_source": [column],  # Retrieve only the specified column
            "query": {"match_all": {}}  # Fetch all documents
        },
        scroll=scroll_time,
    )

    # Extract the scroll ID and hits
    scroll_id = response["_scroll_id"]
    hits = response["hits"]["hits"]

    # Process the initial batch
    values.extend([hit["_source"][column] for hit in hits if column in hit["_source"]])

    # Continue scrolling until no hits are left
    while len(hits) > 0:

        scroll_id = response["_scroll_id"]
        hits = response["hits"]["hits"]
        values.extend([hit["_source"][column] for hit in hits if column in hit["_source"]])
        response = client.scroll(scroll_id=scroll_id, scroll=scroll_time)

    # Clear the scroll context
    client.clear_scroll(scroll_id=scroll_id)

    # Return the collected values
    return values


# if __name__ == '__main__':
#     client = Elasticsearch(constants.DatabaseAddr.CLIENT_ADDR.value, request_timeout=100)
#
#
#     # obtain directories & display content
#     directories = get_column_values_scroll(client=client, index=constants.DatabaseAddr.DB_NAME.value, column="directory")
#     directories = ['Weapons', 'Firearms']
#
#     for dir in directories:
#         display_directory_content(client=client, directory=dir, save_path=constants.Paths.LOCAL_RESULTS_SAVE_PATH.value + 'wordclouds/')
#     #display_directory_content(client=client, directory='Weapons', save_path=constants.Paths.LOCAL_RESULTS_SAVE_PATH.value)
#
#     # # scatter plot of documents highlighting directories
#     # scatter_dir_content(client, save_path=constants.Paths.LOCAL_RESULTS_SAVE_PATH.value)
#     #
#     # # get named entities for documents
#     # nested_field_path = "named_entities"
#     # key_name = "GPE"  #"ORG"
#     # named_entities = get_named_entities_for_docs(client, nested_field_path, key_name)
#     # print(f"Named entities for key '{key_name}': {named_entities}")
#
#     print('finished')
