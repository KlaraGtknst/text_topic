import logging
import tqdm
import data.files as files
from constants import *
from utils.logging_utils import init_debug_config

logger = logging.getLogger(__name__)

if __name__ == '__main__':
    on_server = False
    init_debug_config(log_filename='run_captioning_', on_server=on_server)

    path = Paths.LOCAL_DATA_PATH.value + '/KDE_Projekt/sample_data_server' if (not on_server) \
        else Paths.SERVER_DATA_PATH.value + '/Workshop/'
    num_successes = 0
    limit_num_docs = 2
    paths = files.get_files(path)
    logging.info('Obtained list of paths')
    for path2file in tqdm.tqdm(paths, desc='Extracting text from pdfs'):
        text, success = files.extract_text_from_pdf(path2file)
        num_successes += success

    logging.info(f"Number of successful extractions: {num_successes}/{len(files.get_files(path))}")

