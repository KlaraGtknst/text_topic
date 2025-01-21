import logging
from constants import *
import topic.topic_modeling as tm
import data.files as files
import database.init_elasticsearch as db
import tqdm
import datetime
from data.files import save_sentences_to_file, save_df_to_csv
from utils.logging_utils import init_debug_config
from utils.os_manipulation import exists_or_create

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

    es_db = db.ESDatabase()
    es_db.insert_metadata(src_path=data_path)
    logging.info('Obtained Elasticsearch client and inserted metadata')


    # texts
    pdfs = files.get_files(path=data_path)
    logging.info('Obtained pdfs')
    sentences = []
    for i in tqdm.tqdm(range(len(pdfs)), desc='Extracting text from pdfs'):
        pdf = pdfs[i]
        sentence = files.extract_text_from_pdf(pdf)
        if type(sentence) != str:
            sentence = str(sentence)
        sentences.extend([sentence])
    logging.info('Extracted text from pdfs')
    save_sentences_to_file(sentences=sentences, dataset_path=save_sentences_path,
                           save_filename=f'sentences_ETYNTKE_{date}.txt')
    logging.info('Saved sentences to file')

    if load_existing_topic_model:
        model = tm.TopicModel(documents=None)
        model.load_model(path=model_path, filename='topic_model_01_05_25')
    else:
        model = tm.TopicModel(documents=sentences)
        model.save_model(path=model_path)  # unique name with date

    logging.info('Created and saved topic model')

    # test document-topic incidence
    start = 0
    duration = len(sentences)
    doc_ids = list(range(start, start + len(sentences[start:start + duration]) - 1))

    doc_topic_incidence = model.get_document_topic_incidence(doc_ids=doc_ids)
    save_df_to_csv(df=doc_topic_incidence, path=incidence_save_path, file_name=f"doc_topic_incidence_{date}")
    logging.info("obtained & saved doc-topic incidence")

    # determine optimal threshold for document-topic incidence
    threshold, row_norm_doc_topic_df = model.determine_threshold_doc_topic_threshold(doc_topic_incidence,
                                                                                     opt_density=0.1,
                                                                                     save_path=plot_save_path)
    logging.info(f"optimal threshold: {threshold}")
    thres_row_norm_doc_topic_df = model.apply_threshold_doc_topic_incidence(row_norm_doc_topic_df, threshold=threshold)
    save_df_to_csv(df=thres_row_norm_doc_topic_df, path=incidence_save_path,
                   file_name=f"thres_row_norm_doc_topic_incidence_{date}")
    logging.info("obtained & saved thresholded doc-topic incidence")

    # test term-topic incidence
    term_topic_incidence = model.get_term_topic_incidence(doc_ids=doc_ids)
    save_df_to_csv(term_topic_incidence, incidence_save_path, f"term_topic_incidence_{date}")
    logging.info("obtained & saved term-topic incidence")
