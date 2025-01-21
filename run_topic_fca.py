import constants
from data.files import get_files
from topic.topic_fca import *
from topic.topic_modeling import TopicModel

if __name__ == '__main__':
    on_server = True
    init_debug_config(log_filename='run_topic_fca_', on_server=on_server)
    date = get_date()
    path = constants.Paths.SERVER_DATA_PATH.value + '/Vehicles/' if on_server else (
            constants.Paths.LOCAL_DATA_PATH.value + "/KDE_Projekt/sample_data_server/")
    model_path = constants.Paths.SERVER_PATH_TO_PROJECT.value + 'models/' if on_server else '../models/'
    incidence_save_path = constants.Paths.SERVER_INC_SAVE_PATH.value + '/' + date + '/' if on_server else (
            constants.Paths.LOCAL_DATA_PATH.value + '/incidences/' + date + '/')

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

    dataset_path = constants.Paths.SERVER_PATH_TO_PROJECT.value + 'dataset/' + 'sentences_ETYNTKE01_16_25.txt'
    with open(dataset_path) as f:
        sentences = f.read().splitlines()

    model = TopicModel(documents=sentences)
    topic_fca = TopicFCA()
    # FIXME: doesn't work (top2vec.Top2Vec is no module)
    # model.load_model(path=model_path, filename='01_16_25topic_model_01_17_25' if on_server else 'topic_model')
    topic_fca.obtain_doc_topic_inc_per_subdir(parent_path=path, save_path=incidence_save_path, topic_model=model)
