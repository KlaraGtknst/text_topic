from enum import Enum


class DatabaseAddr(Enum):
    CLIENT_ADDR: str = "http://localhost:9200"
    PUMBAA_CLIENT_ADDR: str = "http://watzmann:9200"    # pumbaa and watzmann are servers, the index is on watzmann
    DB_NAME: str = "txt_db"


class Paths(Enum):  # change the paths to your local/ server paths
    # data paths
    TEST_TRAINING_PATH: str = "/Users/klara/Downloads"
    LOCAL_DATA_PATH: str = "/Users/klara/Downloads"
    SERVER_DATA_PATH: str = "/norgay/bigstore/kgu/data/ETYNTKE"
    # path to project
    SERVER_PATH_TO_PROJECT: str = "/norgay/bigstore/kgu/dev/text_topic/"
    # save paths
    LOCAL_RESULTS_SAVE_PATH: str = "/Users/klara/Developer/Uni/WiSe2425/text_topic/results/"
    SERVER_PLOTS_SAVE_PATH: str = "/norgay/bigstore/kgu/dev/text_topic/results/plots/"
    SERVER_INC_SAVE_PATH: str = "/norgay/bigstore/kgu/dev/text_topic/results/incidences/"
    SERVER_FCA_SAVE_PATH: str = "/norgay/bigstore/kgu/dev/text_topic/results/fca/"
    SERVER_CLJ_RESULTS_PATH: str = "/norgay/bigstore/kgu/dev/clj_exploration_leaks/results/"
    LOCAL_CLJ_RESULTS_PATH: str = "/Users/klara/Developer/Uni/WiSe2425/text_topic/results/"
    # logging
    LOCAL_LOGGING_PATH: str = "/Users/klara/Downloads/logs/"
    SERVER_LOGGING_PATH: str = "/norgay/bigstore/kgu/logs/text_topic/"
