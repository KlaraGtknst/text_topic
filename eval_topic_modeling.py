import logging
from datetime import datetime

import pandas as pd

import database.init_elasticsearch as db
from constants import Paths
from utils.logging_utils import init_debug_config
import tqdm
import data.files as files
import topic.topic_modeling as tm

if __name__ == '__main__':
    date = '01_16_25' #datetime.now().strftime('%x').replace('/', '_')
    on_server = True
    init_debug_config(log_filename='eval_topic_modeling_', on_server=on_server)
    data_path = Paths.SERVER_DATA_PATH.value
    save_sentences_path = Paths.SERVER_PATH_TO_PROJECT.value + 'dataset/'
    model_path = Paths.SERVER_PATH_TO_PROJECT.value + 'models/'
    incidence_save_path = Paths.SERVER_INC_SAVE_PATH.value + '16_01_25/'#date + '/'
    plot_save_path = Paths.SERVER_PLOTS_SAVE_PATH.value + date + '/'

    es_db = db.ESDatabase()
    es_db.insert_metadata(src_path=data_path)
    logging.info('Obtained Elasticsearch client and inserted metadata')
    paths2sentences = {}
    # texts
    file_paths = files.get_files(path=data_path)[:3000]   # TODO: remove this later
    logging.info('Obtained pdfs')
    for file_path in tqdm.tqdm(file_paths, desc='Extracting text from pdfs'):
        sentence = files.extract_text_from_pdf(file_path)
        if type(sentence) != str:
            sentence = str(sentence)
        paths2sentences[file_path] = sentence
    logging.info('Extracted text from pdfs')

    sentences = list(paths2sentences.values())
    model = tm.TopicModel(documents=sentences)
    model.save_model(path=model_path)  # unique name with date
    logging.info('Created and saved topic model')

    # # document-topic incidence
    # start = 0
    # duration = len(sentences)
    # doc_ids = list(range(start, start + len(sentences[start:start + duration]) - 1))
    #
    # doc_topic_incidence = model.get_document_topic_incidence(doc_ids=doc_ids)
    # doc_topic_save_path = incidence_save_path + f"doc_topic_incidence_{date}.csv"
    # files.save_df_to_csv(df=doc_topic_incidence, path=incidence_save_path, file_name=f"doc_topic_incidence_{date}")
    # logging.info("obtained & saved doc-topic incidence to ", doc_topic_save_path)

    # determine optimal threshold for document-topic incidence
    # threshold, row_norm_doc_topic_df = model.determine_threshold_doc_topic_threshold(doc_topic_incidence,
    #                                                                                  opt_density=0.1,
    #                                                                                  save_path=plot_save_path)
    # logging.info(f"optimal threshold: {threshold}")
    # thres_row_norm_doc_topic_df = model.apply_threshold_doc_topic_incidence(row_norm_doc_topic_df, threshold=threshold)
    # files.save_df_to_csv(df=thres_row_norm_doc_topic_df, path=incidence_save_path,
    #                      file_name=f"thres_row_norm_doc_topic_incidence_{date}")
    thres_row_norm_doc_topic_save_path = incidence_save_path + f"doc_topic_incidence_{date}.csv"
    # thres_row_norm_doc_topic_df = pd.read_csv(incidence_save_path + f"thres_row_norm_doc_topic_incidence_{date}", index_col=0)
    logging.info("obtained & saved thresholded doc-topic incidence")

    # test term-topic incidence
    term_topic_save_path = incidence_save_path + f"term_topic_incidence_{date}.csv"
    save_path_topic_words = incidence_save_path + f'topic_words_{date}.json'
    # term_topic_incidence = model.get_term_topic_incidence(doc_ids=doc_ids,
    #                                                       save_path_topic_words=save_path_topic_words)
    # files.save_df_to_csv(term_topic_incidence, incidence_save_path, f"term_topic_incidence_{date}")
    logging.info("obtained & saved term-topic incidence in ", term_topic_save_path)

    average_precision, average_recall = model.evaluate_topic_model(doc_topic_incidence_path=thres_row_norm_doc_topic_save_path,
                                                                   save_path_topic_words=save_path_topic_words,
                                                                   save_df_path=incidence_save_path + f"evaluation_{date}.csv")

    logging.info(f"average precision: {average_precision}, average recall: {average_recall}")
    logging.info("Finished eval_topic_modeling.py")
