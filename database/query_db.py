from constants import *
from database.init_elasticsearch import initialize_db
import topic.topic_modeling as tm
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from visualization.two_d_display import scatter_documents_2d
from utils.os_manipulation import save_or_not

def search_db(client):
    '''
    Returns ten first documents in the database.
    :param client: Elasticsearch client
    :return: result of the query
    '''
    res = client.search(index=DB_NAME, body={
        'size': 10,
        'query': {
            'match_all': {}
        }
    })
    return res

def obtain_directories(client):
    '''
    Returns a set of all directories in the database.
    :param client: Elasticsearch client
    :return: list of directories
    '''
    res = search_db(client)
    return set([r['_source']['directory'] for r in res['hits']['hits']])

def get_directory_content(client, directory:str):
    '''
    Returns a list of all texts in a given directory (only this directory and not its children).
    :param client: Elasticsearch client
    :return: None
    '''
    res = client.search(index=DB_NAME, body={
        '_source': ['text'],
        'query': {
            'match': {
                'directory': directory
            }
        }
    })
    texts = [r['_source']['text'] for r in res['hits']['hits']]
    return texts

def display_directory_content(client, directory:str, save_path=None):
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
    save_or_not(plt, file_name='wordcloud_' + directory + '.png', save_path=save_path, format='svg')
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




if __name__ == '__main__':
    #args = arguments()
    src_path = TEST_TRAINING_PATH#args.directory
    save_dir = SAVE_PATH

    # get client of existing database
    client = initialize_db(src_path, client_addr=CLIENT_ADDR, init_db=False)

    # search for documents in database
    res = search_db(client)
    print('result: ', res)

    # obtain directories & display content
    display_directory_content(client, directory='DB_Tickets', save_path=save_dir)

    # scatter plot of documents highlighting directories
    scatter_dir_content(client, save_path=save_dir)

    print('finished')