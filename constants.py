from enum import Enum


class DatabaseAddr(Enum):
    CLIENT_ADDR: str = "http://localhost:9200"
    PUMBAA_CLIENT_ADDR: str = "http://watzmann:9200"
    DB_NAME: str = "txt_db"


class Paths(Enum):
    TEST_TRAINING_PATH: str = "/Users/klara/Downloads"
    LOCAL_DATA_PATH: str = "/Users/klara/Downloads"
    SERVER_DATA_PATH: str = "/norgay/bigstore/kgu/data/ETYNTKE"
    SERVER_PATH_TO_PROJECT: str = "/norgay/bigstore/kgu/dev/text_topic/"
    LOCAL_RESULTS_SAVE_PATH: str = "/Users/klara/Developer/Uni/WiSe2425/text_topic/results/"
    SERVER_PLOTS_SAVE_PATH: str = "/norgay/bigstore/kgu/dev/text_topic/results/plots/"
    SERVER_INC_SAVE_PATH: str = "/norgay/bigstore/kgu/dev/text_topic/results/plots/results/incidences/"
    SERVER_CLJ_RESULTS_PATH: str = "/norgay/bigstore/kgu/dev/clj_exploration_leaks/results/"
    LOCAL_CLJ_RESULTS_PATH: str = "/Users/klara/Developer/Uni/WiSe2425/text_topic/results/"
    LOCAL_LOGGING_PATH: str = "/Users/klara/Downloads/logs/"
    SERVER_LOGGING_PATH: str = "/norgay/bigstore/kgu/logs/text_topic/"
