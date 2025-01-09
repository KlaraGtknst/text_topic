from elasticsearch import Elasticsearch
from visualization.two_d_display import scatter_documents_2d
import constants

if __name__ == '__main__':

    # 2D scatter plot of the documents colored by their parent directory
    client = Elasticsearch(constants.CLIENT_ADDR)
    scatter_documents_2d(client, save_path=constants.SERVER_SAVE_PATH, use_tsne=True)
