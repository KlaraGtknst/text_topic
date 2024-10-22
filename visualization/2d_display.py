import matplotlib.pyplot as plt
import seaborn as sns
from elasticsearch import Elasticsearch
from constants import *
from sklearn.decomposition import PCA
import pandas as pd


def scatter_documents_2d(client):
    '''
    This function creates a 2D scatter plot of the documents.
    '''
    # obtain results from elastic search
    res = client.search(index=DB_NAME, body={
        'size': 10,
        'query': {
            'match_all': {}
        }
    })
    embeddings = [r['_source']['embedding'] for r in res['hits']['hits']]
    class_dirs = [r['_source']['directory'] for r in res['hits']['hits']]

    # reduce dimensionality to 2D
    pca = PCA(n_components=2)
    transformed_embs = pca.fit_transform(embeddings)

    # create dataframe
    df = pd.DataFrame({'x': transformed_embs[:, 0], 'y': transformed_embs[:, 1], 'parent directory': class_dirs})

    # plot highlighting the different directories as classes
    fig = plt.figure()
    sns.scatterplot(x=df['x'], y=df['y'], hue=df['parent directory'])
    plt.title('2D scatter plot of the documents')
    plt.legend(bbox_to_anchor=(1.04, 1), loc="upper left")
    fig.tight_layout()
    plt.show()



if __name__ == '__main__':
    client = Elasticsearch(CLIENT_ADDR)
    scatter_documents_2d(client)

