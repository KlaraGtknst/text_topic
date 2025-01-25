import constants
from data.files import get_files
from database.query_db import get_texts_from_docs
from topic.topic_fca import *
from topic.topic_modeling import TopicModel
import database.init_elasticsearch as db

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    on_server = True
    init_debug_config(log_filename='run_topic_fca_', on_server=on_server)
    date = get_date()
    path = constants.Paths.SERVER_DATA_PATH.value if on_server else (    # TODO: omit later: + '/Vehicles/'  + '/Firearms/'
            constants.Paths.LOCAL_DATA_PATH.value + "/KDE_Projekt/sample_data_server/")
    model_path = constants.Paths.SERVER_PATH_TO_PROJECT.value + 'models/' if on_server else '../models/'
    incidence_save_path = constants.Paths.SERVER_INC_SAVE_PATH.value + date + '/' if on_server else (
            constants.Paths.LOCAL_DATA_PATH.value + '/incidences/' + date + '/')
    fca_save_path = constants.Paths.SERVER_FCA_SAVE_PATH.value + date + '/' if on_server else (
            constants.Paths.LOCAL_DATA_PATH.value + '/fca_res/' + date + '/')

    # date = "01_08_25"
    # save_date = "01_14_25_02"
    # path = constants.SERVER_PATH if on_server else "/Users/klara/Documents/uni/"
    # dataset_path = constants.SERVER_PATH_TO_PROJECT + 'dataset/' if on_server else "../dataset/"
    # model_path = constants.SERVER_PATH_TO_PROJECT + 'models/' if on_server else '../models/'
    # incidence_save_path = constants.SERVER_PATH_TO_PROJECT + 'results/incidences/server_080125/' \
    #     if on_server else "../results/incidences/"
    # plot_save_path = constants.SERVER_PATH_TO_PROJECT + 'results/plots/server_080125/' \
    #     if on_server else "../results/plots/"
    # top_doc_filename = f"thres_row_norm_doc_topic_incidence{date}.csv" \
    #     if on_server else "thres_row_norm_doc_topic_incidence.csv"
    # term_topic_filename = f"term_topic_incidence{date}.csv" \
    #     if on_server else "term_topic_incidence.csv"
    #
    # # Load the doc-topic context
    # print("Starting to load doc-topic context as fimi to path: ", incidence_save_path)
    # doc_topic_ctx = csv2ctx(path_to_file=incidence_save_path, filename=top_doc_filename, prefix="doc_")
    # ctx2fimi(doc_topic_ctx, path_to_file=incidence_save_path, filename=f"doc_topic_fimi_{save_date}", prefix="doc_")
    # print("Doc-topic context loaded and saved as fimi to path: ", incidence_save_path)
    # print("--------------------------")
    #
    # # Load the term-topic context
    # print("Starting to load term-topic context as fimi to path: ", incidence_save_path)
    # term_topic_ctx = csv2ctx(path_to_file=incidence_save_path, filename=term_topic_filename, prefix="term_")
    # ctx2fimi(term_topic_ctx, path_to_file=incidence_save_path, filename=f"term_topic_fimi_{save_date}", prefix="term_")
    # print("Term-topic context loaded and saved as fimi to path: ", incidence_save_path)
    #
    # # convert term-topic fimi to rows of integers representing topics incl. mapping as edn file
    # path2fimi = incidence_save_path + f"term_topic_fimi_{save_date}"
    # topics2integers(path2fimi=path2fimi + ".fimi", save_path=path2fimi + "_topics_as_integers.fimi")
    #
    # # TODO: obtain intents efficiently via pcbo (terminal)

    # run on pumbaa bc captioner
    es_db = db.ESDatabase(client_addr=constants.DatabaseAddr.PUMBAA_CLIENT_ADDR.value)
    logging.info("Obtained Elasticsearch client")

    sentences = get_texts_from_docs(client=es_db.get_es_client())
    logging.info(f"Loaded {len(sentences)} sentences.")

    model = TopicModel(documents=sentences)   # TODO: omit later [:500]
    logging.info("Obtained topic model")

    topic_fca = TopicFCA()
    logging.info("Obtained topic fca instance")

    if not path.endswith('/'):
        path = path + '/'

    for current_directory, subdirectories, files in os.walk(path):
        for sub_dir in subdirectories:
            logging.info(f"Starting to obtain doc-topic incidence for subdirectory {sub_dir}")
            if sub_dir in ['Firearm Manuals', 'Law (re Firearms)', 'Machine Guns', 'Firearms', '12-volt Guide - Junk Science', 'Camouflage and Concealment', 'Home Power Magazine', 'Marksmanship and Sniping', 'Energy and Fuel', 'Issues 1 - 42', 'Robotics', 'Exotic Weapons', 'ng & Caching', 'Knives, Swords']:
                continue
            topic_fca.obtain_doc_topic_inc_per_subdir(parent_path=path + sub_dir + '/', save_path=fca_save_path,
                                                      topic_model=model, recursive=False)
            logging.info(f"Finished obtaining doc-topic incidence for subdirectory {sub_dir}")

    logging.info("The end")
