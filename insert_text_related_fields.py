import logging
import constants
import database.init_elasticsearch as db
from utils.logging_utils import init_debug_config

logger = logging.getLogger(__name__)

if __name__ == '__main__':  # run this code on pumbaa server, bc GPU is needed
    on_server = True
    init_debug_config(log_filename='insert_text_related_fields_', on_server=on_server)

    # initialize Elasticsearch client
    es_db = db.ESDatabase(client_addr=constants.DatabaseAddr.PUMBAA_CLIENT_ADDR.value)

    # insert text related fields
    es_db.insert_text_related_fields_bulk(src_path=constants.Paths.SERVER_DATA_PATH.value)
    logging.info('Finished inserting text related fields: text, embedding, named_entities')
