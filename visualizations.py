from elasticsearch import Elasticsearch

from data import files
from visualization.two_d_display import scatter_documents_2d
from visualization.visualize_stats import stats_as_bar_charts
from visualization.visualize_named_entity_clusters import display_NE_cluster
import constants
import datetime
import tqdm

if __name__ == '__main__':
    date = datetime.datetime.now().strftime('%x').replace('/', '_')

    # 2D scatter plot of the documents colored by their parent directory
    # client = Elasticsearch(constants.CLIENT_ADDR)
    # scatter_documents_2d(client, save_path=constants.SERVER_SAVE_PATH, reducer='UMAP', unique_id_suffix=date)
    # scatter_documents_2d(client, save_path=constants.SERVER_SAVE_PATH, reducer='TSNE', unique_id_suffix=date)
    # scatter_documents_2d(client, save_path=constants.SERVER_SAVE_PATH, reducer='PCA', unique_id_suffix=date)

    # # visualize data stats
    # base_path2csv = constants.SERVER_STATS_PATH
    # csv_files = files.get_files(path=base_path2csv, file_type='csv')
    # for i in tqdm.tqdm(range(len(csv_files)), desc='Producing bar charts of statistic files'):
    #     csv_file = csv_files[i]
    #     stats_as_bar_charts(path2csv=csv_file, save_path=constants.SERVER_SAVE_PATH,
    #                         unique_id_suffix=date + '_' + str(i))

    # visualize named entity clusters
    path = "/norgay/bigstore/kgu/dev/text_topic/results/plots/server_080125/cluster_NER/cluster_NE_results_PERSON_01_10_25.json"
    ne_cluster_dict = files.load_json(path)
    display_NE_cluster(ne_results=ne_cluster_dict, reducer="PCA", category="PERSON", save=True)
