import logging
import tqdm
import constants
from data import files
from database.init_elasticsearch import ESDatabase
from utils.logging_utils import get_date, init_debug_config
import utils.os_manipulation as osm
from visualization.two_d_display import scatter_documents_2d
from visualization.vis_fca import display_context
from visualization.visualize_named_entity_clusters import display_NE_cluster
from visualization.visualize_stats import stats_as_bar_charts

if __name__ == '__main__':
    on_server = True
    init_debug_config(log_filename='visualizations_', on_server=on_server)
    date = get_date()

    # visualize data stats
    # base_path2csv = constants.Paths.SERVER_CLJ_RESULTS_PATH.value if on_server \
    #     else constants.Paths.LOCAL_CLJ_RESULTS_PATH.value
    # csv_files = files.get_files(path=base_path2csv, file_type='csv')
    # for i in tqdm.tqdm(range(len(csv_files)), desc='Producing bar charts of statistic files'):
    #     csv_file = csv_files[i]
    #     logging.info(f"Started with csv file: {csv_file}")
    #     stats_as_bar_charts(path2csv=csv_file, save_path=constants.Paths.SERVER_PLOTS_SAVE_PATH.value + get_date(),
    #                         unique_id_suffix=get_date() + '_' + str(i))
    #
    # # 2D scatter plot of the documents colored by their parent directory
    # es_db = ESDatabase()
    # client = es_db.get_es_client()
    # scatter_documents_2d(client, save_path=constants.Paths.SERVER_PLOTS_SAVE_PATH.value + get_date(), reducer='UMAP', unique_id_suffix=date)
    # scatter_documents_2d(client, save_path=constants.Paths.SERVER_PLOTS_SAVE_PATH.value + get_date(), reducer='TSNE', unique_id_suffix=date)
    # scatter_documents_2d(client, save_path=constants.Paths.SERVER_PLOTS_SAVE_PATH.value + get_date(), reducer='PCA', unique_id_suffix=date)

    # visualize named entity clusters
    # save_path = constants.Paths.SERVER_PLOTS_SAVE_PATH.value + get_date() + "/cluster_NER/"
    # json_files = files.get_files(path=constants.Paths.SERVER_PLOTS_SAVE_PATH.value + "/cluster_NER/", file_type='json')
    # for reducer in ['TSNE', 'PCA', 'UMAP']:
    #     logging.info(f"Started with reducer: {reducer}")
    #     for file_path in tqdm.tqdm(json_files, desc='Producing plots of NER cluster files'):
    #         category = file_path.split('/')[-1].split('_')[3]
    #         ne_cluster_dict = files.load_dict_from_json(path=file_path)
    #         display_NE_cluster(ne_results=ne_cluster_dict, reducer=reducer, category=category,
    #                            save_path=save_path)

    # visualize FCA contexts
    path2across_dir_csv = "/norgay/bigstore/kgu/dev/clj_exploration_leaks/results/fca-dir-concepts/across-dir/" if (
        on_server) else "/Users/klara/Developer/Uni/WiSe2425/clj_exploration_leaks/results/fca-dir-concepts/across-dir/"
    save_path = constants.Paths.SERVER_FCA_SAVE_PATH.value + date + '/' if on_server else \
        f"/Users/klara/Developer/Uni/WiSe2425/text_topic/results/fca/{date}/"
    filename_of_csv = "server-across-dir-incidence-matrix.csv"  # "across-dir-incidence-matrix.csv"

    # across-dir-incidence-matrix
    # osm.exists_or_create(path=save_path)
    # display_context(path2csv=path2across_dir_csv, save_path=save_path, filename_of_csv=filename_of_csv, on_server=on_server)

    # single-dir-incidence-matrix
    osm.exists_or_create(path=save_path + 'single_dir_contexts/')
    if on_server:
        path2single_csv = "/norgay/bigstore/kgu/dev/text_topic/results/fca/01_27_25/"
        for dir in files.get_files(path=path2single_csv, file_type='csv', recursive=False):
            display_context(path2csv=dir, save_path=save_path + 'single_dir_contexts/',
                            filename_of_csv=dir.split('/')[-1], on_server=on_server)

    logging.info('Finished visualizations')
