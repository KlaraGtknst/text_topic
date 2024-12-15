from constants import *
from database.init_elasticsearch import initialize_db
import topic.topic_modeling as tm
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from visualization.two_d_display import scatter_documents_2d
from utils.os_manipulation import save_or_not


def search_db(client, num_res: int = 10):
    '''
    Returns ten first documents in the database.
    :param client: Elasticsearch client
    :return: result of the query
    '''
    res = client.search(index=DB_NAME, body={
        'size': num_res,
        'query': {
            'match_all': {}
        }
    })
    return res


def get_num_indexed_documents(client):
    '''
    Returns the number of documents in the database.
    :param client: Elasticsearch client
    :return: number of documents
    '''
    client.indices.refresh(index=DB_NAME)
    count = int(client.cat.count(index=DB_NAME, format="json")[0]["count"])
    return count


def obtain_directories(client):
    '''
    Returns a set of all directories in the database.
    :param client: Elasticsearch client
    :return: list of directories
    '''
    count = get_num_indexed_documents(client)
    res = search_db(client, num_res=count)
    return set([r['_source']['directory'] for r in res['hits']['hits']])


def get_directory_content(client, directory: str):
    '''
    Returns a list of all texts in a given directory (only this directory and not its children).
    :param client: Elasticsearch client
    :return: None
    '''
    res = client.search(index=DB_NAME, body={
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


def display_directory_content(client, directory: str, save_path=None):
    '''
    Displays a wordcloud of the content of a given directory.
    If save_path is not None, saves the wordcloud as a .png file.
    :param client: Elasticsearch client
    :return: None
    '''
    sentences = get_directory_content(client, directory)
    text = ' '.join(sentences)
    wordcloud = WordCloud(max_font_size=40).generate(text)
    plt.figure()
    plt.title('Wordcloud of directory: ' + directory)
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    save_or_not(plt, file_name='wordcloud_' + directory + '.svg', save_path=save_path, format='svg')
    plt.show()


def scatter_dir_content(client, save_path=None):
    '''
    Displays a 2D scatter plot of the documents in the database.
    The documents are represented by their embeddings.
    Each document is colored according to its parent directory.
    The plot is saved as a .png file if save_path is not None.
    :param client: Elasticsearch client
    :return: None
    '''
    scatter_documents_2d(client, save_path=save_path)


def get_named_entities_for_doc(client, nested_field_path, key_name, num_res: int = 10):
    '''
    Returns ten first documents in the database.
    :param client: Elasticsearch client
    :param nested_field_path: Path to the nested field (e.g., "parent.child").
    :param key_name: The specific key to retrieve values for.
    :return: result of the query
    '''
    # FIXME: implement query for named entity recognition -> returns empty list
    query = {
        "query": {
            "nested": {
                "path": nested_field_path,
                "query": {
                    "exists": {
                        "field": f"{nested_field_path}.{key_name}"  # Check if the 'value' field exists in the nested field
                    }
                },
            }
        }
    }
    # query = {
    #     "query": {
    #         "exists": {
    #             "field": nested_field_path
    #         }
    #     },
    #     "_source": nested_field_path
    # }

    # Execute the search query
    response = client.search(index=DB_NAME, body=query, size=1000)
    print('response: ', response)

    # Extract the values
    values = []
    for hit in response['hits']['hits']:
        # Navigate to the nested field
        nested_field = hit['_source']
        for part in nested_field_path.split('.'):
            nested_field = nested_field.get(part, {})

        # Add the value for the specified key if it exists
        if key_name in nested_field:
            values.append(nested_field[key_name])

    return values


if __name__ == '__main__':
    #args = arguments()
    src_path = TEST_TRAINING_PATH  #args.directory
    save_dir = SAVE_PATH + '/plots/'

    # get client of existing database
    client = initialize_db(src_path=src_path, client_addr=CLIENT_ADDR, create_db=False)

    # search for documents in database
    #res = search_db(client)
    #print('result: ', res)

    # obtain directories & display content
    # display_directory_content(client, directory='SozNet', save_path=save_dir)
    #
    # # scatter plot of documents highlighting directories
    # scatter_dir_content(client, save_path=save_dir)

    # get named entities for documents
    nested_field_path = "named_entities"
    key_name = "GPE"#"ORG"
    named_entities = get_named_entities_for_doc(client, nested_field_path, key_name)
    print(f"Named entities for key '{key_name}': {named_entities}")

    print('finished')
