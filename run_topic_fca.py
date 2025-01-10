import constants
from topic.topic_fca import *


if __name__ == '__main__':
    on_server = True
    date = "01_08_25"
    path = constants.SERVER_PATH if on_server else "/Users/klara/Documents/uni/"
    dataset_path = constants.SERVER_PATH_TO_PROJECT + 'dataset/' if on_server else "../dataset/"
    model_path = constants.SERVER_PATH_TO_PROJECT + 'models/' if on_server else '../models/'
    incidence_save_path = constants.SERVER_PATH_TO_PROJECT + 'results/incidences/server_080125/' \
        if on_server else "../results/incidences/"
    plot_save_path = constants.SERVER_PATH_TO_PROJECT + 'results/plots/server_080125/' \
        if on_server else "../results/plots/"
    top_doc_filename = f"thres_row_norm_doc_topic_incidence{date}.csv" \
        if on_server else "thres_row_norm_doc_topic_incidence.csv"
    term_topic_filename = f"term_topic_incidence{date}.csv" \
        if on_server else "term_topic_incidence.csv"

    # Load the doc-topic context
    doc_topic_ctx = csv2ctx(path_to_file=incidence_save_path, filename=top_doc_filename)
    ctx2fimi(doc_topic_ctx, path_to_file=incidence_save_path)

    # Load the term-topic context
    term_topic_ctx = csv2ctx(path_to_file=incidence_save_path, filename=term_topic_filename)
    ctx2fimi(term_topic_ctx, path_to_file=incidence_save_path)

    # TODO: obtain intents efficiently via pcbo (terminal)