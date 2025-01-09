import matplotlib.pyplot as plt
import seaborn as sns
from elasticsearch import Elasticsearch
from constants import *
from sklearn.decomposition import PCA
import pandas as pd
from utils.os_manipulation import save_or_not


def scatter_documents_2d(client, save_path=None):
    '''
    This function creates a 2D scatter plot of the documents.
    The documents are represented by their embeddings.
    Each document is colored according to its parent directory.
    The plot is saved as a .png file if save_path is not None.
    :param client: Elasticsearch client
    :param save_path: path to save the plot
    :return -
    '''
    # obtain results from elastic search
    client.indices.refresh(index=DB_NAME)
    count = int(client.cat.count(index=DB_NAME, format="json")[0]["count"])
    res = client.search(index=DB_NAME, body={
        'size': count,
        'query': {
            'match_all': {}
        }
    })

    embeddings = [r['_source']['embedding'] for r in res['hits']['hits']]
    class_dirs = [r['_source']['directory'] for r in res['hits']['hits']]

    # reduce dimensionality to 2D
    pca = PCA(n_components=2)   # TODO: TSNE/ UMAP instead of PCA?
    transformed_embs = pca.fit_transform(embeddings)

    # create dataframe
    df = pd.DataFrame({'x': transformed_embs[:, 0], 'y': transformed_embs[:, 1], 'parent directory': class_dirs})

    # plot highlighting the different directories as classes
    fig = plt.figure(figsize=(12, 8))
    sns.scatterplot(x=df['x'], y=df['y'], hue=df['parent directory'])
    plt.title('2D scatter plot of the documents')
    plt.legend(loc="upper left", fontsize="9", fancybox=True, shadow=True, ncol=3, bbox_to_anchor=(1.04, 1))
    #legend.set_in_layout(False)
    fig.tight_layout()
    save_or_not(plt, file_name='scatter_documents_dir_2d.png', save_path=save_path, format='svg')
    plt.show()



# if __name__ == '__main__':
#     client = Elasticsearch(CLIENT_ADDR)
#     scatter_documents_2d(client, save_path=SERVER_SAVE_PATH)

