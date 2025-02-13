import datetime
import logging
import tqdm
import data.files as files
import database.init_elasticsearch as db
import topic.topic_modeling as tm
from constants import *
from data.files import save_sentences_to_file, save_df_to_csv
from utils.logging_utils import init_debug_config

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    date = datetime.datetime.now().strftime('%x').replace('/', '_')
    on_server = True
    init_debug_config(log_filename='main_', on_server=on_server)
    data_path = Paths.SERVER_DATA_PATH.value
    save_sentences_path = Paths.SERVER_PATH_TO_PROJECT.value + 'dataset/'
    model_path = Paths.SERVER_PATH_TO_PROJECT.value + 'models/'
    incidence_save_path = Paths.SERVER_INC_SAVE_PATH.value + date + '/'
    plot_save_path = Paths.SERVER_PLOTS_SAVE_PATH.value + date + '/'
    load_existing_topic_model = False

    # initialize Elasticsearch client
    es_db = db.ESDatabase()
    es_db.init_db()  # use: initialize_db if you want to delete the old index
    logging.info('Initialized Elasticsearch client')