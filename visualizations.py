from elasticsearch import Elasticsearch
from visualization.two_d_display import scatter_documents_2d
import constants
import datetime

if __name__ == '__main__':
    date = datetime.datetime.now().strftime('%x').replace('/', '_')

    # 2D scatter plot of the documents colored by their parent directory
    client = Elasticsearch(constants.CLIENT_ADDR)
    scatter_documents_2d(client, save_path=constants.SERVER_SAVE_PATH, use_tsne=True, unique_id_suffix=date)
    scatter_documents_2d(client, save_path=constants.SERVER_SAVE_PATH, use_tsne=False, unique_id_suffix=date)
