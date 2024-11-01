from constants import *
from database.init_elasticsearch import initialize_db
import topic.topic_modeling as tm
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from visualization.two_d_display import scatter_documents_2d
from utils.os_manipulation import save_or_not

def search_db(client):
    '''
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
    :param client: Elasticsearch client
    :return: list of directories
    '''
    res = search_db(client)
    return set([r['_source']['directory'] for r in res['hits']['hits']])

def get_directory_content(client, directory):
    '''
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

def display_directory_content(client, directory, save_path=None):
    '''
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