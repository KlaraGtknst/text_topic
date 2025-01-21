import logging
import os
from glob import glob

from elasticsearch import Elasticsearch

from data import files
from data.caption_images import ImageCaptioner
from utils.logging_utils import get_date, init_debug_config
from utils.os_manipulation import exists_or_create
from visualization.two_d_display import scatter_documents_2d
from visualization.visualize_stats import stats_as_bar_charts
from visualization.visualize_named_entity_clusters import display_NE_cluster
import constants
import datetime
import tqdm

if __name__ == '__main__':
    on_server = True
    init_debug_config(log_filename='visualizations_', on_server=on_server)

    # 2D scatter plot of the documents colored by their parent directory
    # client = Elasticsearch(constants.CLIENT_ADDR)
    # scatter_documents_2d(client, save_path=constants.Paths.SERVER_PLOTS_SAVE_PATH.value + get_date(), reducer='UMAP', unique_id_suffix=date)
    # scatter_documents_2d(client, save_path=constants.Paths.SERVER_PLOTS_SAVE_PATH.value + get_date(), reducer='TSNE', unique_id_suffix=date)
    # scatter_documents_2d(client, save_path=constants.Paths.SERVER_PLOTS_SAVE_PATH.value + get_date(), reducer='PCA', unique_id_suffix=date)

    # visualize data stats
    base_path2csv = constants.Paths.SERVER_CLJ_RESULTS_PATH.value
    csv_files = files.get_files(path=base_path2csv, file_type='csv')
    for i in tqdm.tqdm(range(len(csv_files)), desc='Producing bar charts of statistic files'):
        csv_file = csv_files[i]
        logging.info(f"Started with csv file: {csv_file}")
        stats_as_bar_charts(path2csv=csv_file, save_path=constants.Paths.SERVER_PLOTS_SAVE_PATH.value + get_date(),
                            unique_id_suffix=get_date() + '_' + str(i))

    # visualize named entity clusters
    # save_path = constants.Paths.SERVER_PLOTS_SAVE_PATH.value + get_date() + "/cluster_NER/"
    # json_files = files.get_files(path=save_path, file_type='json')
    # for reducer in ['TSNE', 'PCA', 'UMAP']:
    #     logging.info(f"Started with reducer: {reducer}")
    #     for i in tqdm.tqdm(range(len(json_files)), desc='Producing plots of NER cluster files'):
    #         file_path = json_files[i]
    #         category = file_path.split('/')[-1].split('_')[3]
    #         ne_cluster_dict = files.load_dict_from_json(path=file_path)
    #         display_NE_cluster(ne_results=ne_cluster_dict, reducer=reducer, category=category,
    #                            save_path=save_path)
