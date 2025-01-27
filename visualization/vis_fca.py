import logging

import constants
import utils.logging_utils as logging_utils
import utils.os_manipulation as osm
import data.files as files
from topic.topic_fca import TopicFCA

logger = logging.getLogger(__name__)


def display_context(path2csv: str, save_path: str, filename_of_csv: str, on_server: bool = False):
    """
    This function displays the context as a graph and saves it.
    :param path2csv: Path to the csv file that contains the context
    :param save_path: Path to the directory where the graph should be saved. Should end with '/'
    :param filename_of_csv: Filename of the csv file that contains the context
    :param on_server: Boolean indicating whether the code is running on the server or locally
    :return: -
    """
    if "thres" in filename_of_csv:
        if ("translated" in filename_of_csv) or ("term" in filename_of_csv):
            return

        if not save_path.endswith('/'):
            save_path = save_path + '/'
        topic_fca = TopicFCA(on_server=on_server)
        osm.exists_or_create(path=save_path)

        # Load the context
        # if not on server -> likely to be across-dir-incidence-matrix -> needs space, hence strip prefix
        # else -> likely to be single-dir-incidence-matrix -> no need to strip prefix
        ctx = topic_fca.csv2ctx(path_to_file=path2csv, filename=filename_of_csv, strip_prefix=(not on_server))
        ctx.lattice.graphviz(view=(not on_server), filename=save_path + f"fca_graph_{logging_utils.get_date()}", format='svg',
                             directory=save_path)


# if __name__ == "__main__":
#     on_server = False
#     date = logging_utils.get_date()
#     # logging_utils.init_debug_config(log_filename='vis_fca_', on_server=on_server)
#     path2across_dir_csv = "/norgay/bigstore/kgu/dev/clj_exploration_leaks/results/fca-dir-concepts/across-dir/" if (
#         on_server) else "/Users/klara/Developer/Uni/WiSe2425/clj_exploration_leaks/results/fca-dir-concepts/across-dir/"
#     save_path = constants.Paths.SERVER_FCA_SAVE_PATH.value + date + '/' if on_server else \
#         f"/Users/klara/Developer/Uni/WiSe2425/text_topic/results/fca/{date}/"
#     filename_of_csv = "server-across-dir-incidence-matrix.csv"  # "across-dir-incidence-matrix.csv"
#
#     # across-dir-incidence-matrix
#     osm.exists_or_create(path=save_path)
#     display_context(path2csv=path2across_dir_csv, save_path=save_path, filename_of_csv=filename_of_csv)
#
#     # single-dir-incidence-matrix
#     osm.exists_or_create(path=save_path + 'single_dir_contexts/')
#     if on_server:
#         path2single_csv = "/norgay/bigstore/kgu/dev/text_topic/results/fca/01_27_25/"
#         for dir in files.get_files(path=path2single_csv, file_type='csv', recursive=False):
#             display_context(path2csv=dir, save_path=save_path + 'single_dir_contexts/',
#                             filename_of_csv=dir.split('/')[-1])

