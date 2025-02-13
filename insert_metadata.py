import logging
import database.init_elasticsearch as db
from constants import *
from utils.logging_utils import init_debug_config

logger = logging.getLogger(__name__)

if __name__ == '__main__':  # run code on watzmann server
    on_server = True
    init_debug_config(log_filename='insert_metadata_', on_server=on_server)

    # initialize Elasticsearch client
    es_db = db.ESDatabase()

    # insert metadata
    es_db.insert_metadata(src_path=Paths.SERVER_DATA_PATH.value)
    logging.info('Obtained Elasticsearch client and inserted metadata')