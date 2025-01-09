import matplotlib.pyplot as plt
import seaborn as sns
from elasticsearch import Elasticsearch
from constants import *
from sklearn.decomposition import PCA
import pandas as pd
from utils.os_manipulation import save_or_not
from scipy.manifold import TSNE

def scatter_documents_2d(client, save_path=None, on_server=False):
    '''
    This function creates a 2D scatter plot of the documents.
    The documents are represented by their embeddings.
    Each document is colored according to its parent directory.
    The plot is saved as a .png file if save_path is not None.
    :param client: Elasticsearch client
    :param save_path: path to save the plot
    :param on_server: if the function is run on the server. Since there are deeper directories on the server,
    the documents are coloured according to the uppermost directory.
    :return -
    '''
    # obtain results from elastic search
    upper_request_limit = 10000
    client.indices.refresh(index=DB_NAME)
    count = int(client.cat.count(index=DB_NAME, format="json")[0]["count"])
    results = []

    res = client.search(index=DB_NAME, body={
        'size': upper_request_limit,  # Number of documents to fetch
        'query': {
            'match_all': {}
        }
    },
                            scroll='2m'  # Keep the scroll context alive for 2 minutes -> used for big data
                            )
    # Get the first batch of results
    scroll_id = res['_scroll_id']
    results.extend(res['hits']['hits'])

    # Fetch additional batches
    while True:
        response = client.scroll(
            scroll_id=scroll_id,
            scroll='2m'
        )
        hits = response['hits']['hits']
        if not hits:  # Break if no more results
            break
        results.extend(hits)  # Add the new results to the list
        scroll_id = response['_scroll_id']  # Update scroll ID for the next iteration

    # Clear the scroll context
    client.clear_scroll(scroll_id=scroll_id)

    embeddings = [r['_source']['embedding'] for r in results]
    class_dirs = [r['_source']['directory'] for r in results]
    # uppermost directory if on server else the directory
    colour_criteria = [r['_source']['path'].split('/ETYNTKE/')[-1].split('/')[0] for r in results] if on_server else class_dirs

    # reduce dimensionality to 2D
    pca = PCA(n_components=2)  # TODO: TSNE/ UMAP instead of PCA?
    tsne = TSNE(n_components=2)
    transformed_embs = pca.fit_transform(embeddings)

    # create dataframe
    df = pd.DataFrame({'x': transformed_embs[:, 0], 'y': transformed_embs[:, 1], 'parent directory': colour_criteria})

    # plot highlighting the different directories as classes
    fig = plt.figure(figsize=(12, 8))
    sns.scatterplot(x=df['x'], y=df['y'], hue=df['parent directory'])
    plt.title('2D scatter plot of the documents')
    plt.legend(loc="upper left", fontsize="9", fancybox=True, shadow=True, ncol=3, bbox_to_anchor=(1.04, 1))
    fig.tight_layout()  # if too many classes: will rise a warning, but the plot will be saved correctly
    save_or_not(plt, file_name='scatter_documents_dir_2d.png', save_path=save_path, format='svg')
    plt.show()

# if __name__ == '__main__':
#     client = Elasticsearch(CLIENT_ADDR)
#     scatter_documents_2d(client, save_path=SERVER_SAVE_PATH)
