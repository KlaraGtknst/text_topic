import matplotlib.pyplot as plt
from elasticsearch import Elasticsearch
from constants import *
from sklearn.decomposition import PCA


def scatter_documents_2d(client):
    '''
    This function creates a 2D scatter plot of the documents.
    '''
    res = client.search(index=DB_NAME, body={
        'size': 10,
        'query': {
            'match_all': {}
        }
    })

    embeddings = [r['_source']['embedding'] for r in res['hits']['hits']]
    print(embeddings)

    pca = PCA(n_components=2)
    transformed_embs = pca.fit_transform(embeddings)
    plt.scatter(transformed_embs[:, 0], transformed_embs[:, 1])
    plt.show()



if __name__ == '__main__':
    client = Elasticsearch(CLIENT_ADDR)
    scatter_documents_2d(client)
    #plt.show()

