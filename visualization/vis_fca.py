import logging

import utils.logging_utils as logging_utils
import utils.os_manipulation as osm
from topic.topic_fca import TopicFCA

logger = logging.getLogger(__name__)


def display_context(path2csv: str, save_path: str, filename_of_csv: str):
    """
    This function displays the context as a graph and saves it.
    :param path2csv: Path to the csv file that contains the context
    :param save_path: Path to the directory where the graph should be saved. Should end with '/'
    :param filename_of_csv: Filename of the csv file that contains the context
    :return: -
    """
    topic_fca = TopicFCA(on_server=False)
    osm.exists_or_create(path=save_path)

    # Load the context
    ctx = topic_fca.csv2ctx(path_to_file=path2csv, filename=filename_of_csv, strip_prefix=True)
    ctx.lattice.graphviz(view=True, filename=save_path + f"fca_graph_{logging_utils.get_date()}", format='svg',
                         directory=save_path)


if __name__ == "__main__":
    on_server = False
    date = logging_utils.get_date()
    # logging_utils.init_debug_config(log_filename='vis_fca_', on_server=on_server)
    path2csv = "/Users/klara/Developer/Uni/WiSe2425/clj_exploration_leaks/results/fca-dir-concepts/across-dir/"
    save_path = f"/Users/klara/Developer/Uni/WiSe2425/text_topic/results/fca/{date}/"
    filename_of_csv = "across-dir-incidence-matrix.csv"
    display_context(path2csv=path2csv, save_path=save_path, filename_of_csv=filename_of_csv)
