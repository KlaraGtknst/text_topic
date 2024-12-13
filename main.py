import constants
import topic.topic_modeling as tm
import data.files as files
import tqdm

from data.files import save_sentences_to_file, save_df_to_csv

if __name__ == '__main__':
    path = constants.SERVER_PATH
    dataset_path = "../dataset/"
    model_path = '../models/'
    incidence_save_path = "../results/incidences/"
    plot_save_path = "../results/plots/"

    # texts
    pdfs = files.get_files(path=path)
    sentences = []
    for i in tqdm.tqdm(range(len(pdfs)), desc='Extracting text from pdfs'):
        pdf = pdfs[i]
        sentence = files.extract_text_from_pdf(pdf)
        if type(sentence) != str:
            sentence = str(sentence)
        sentences.extend([sentence])
    save_sentences_to_file(sentences, dataset_path)

    model = tm.TopicModel(documents=sentences)
    model.save_model(path=model_path)  # unique name with date

    # test document-topic incidence
    start = 0
    duration = len(sentences)
    doc_ids = list(range(start, start + len(sentences[start:start + duration]) - 1))

    doc_topic_incidence = model.get_document_topic_incidence(doc_ids=doc_ids)
    save_df_to_csv(doc_topic_incidence, incidence_save_path, "doc_topic_incidence")
    print("first 5doc-topic incidence:\n", doc_topic_incidence.head())

    # determine optimal threshold for document-topic incidence
    threshold, row_norm_doc_topic_df = model.determine_threshold_doc_topic_threshold(doc_topic_incidence,
                                                                                     opt_density=0.1,
                                                                                     save_path=plot_save_path)
    print("optimal threshold: ", threshold)
    thres_row_norm_doc_topic_df = model.apply_threshold_doc_topic_incidence(row_norm_doc_topic_df, threshold=threshold)
    save_df_to_csv(thres_row_norm_doc_topic_df, incidence_save_path, "thres_row_norm_doc_topic_incidence")
    print("first 5 thresholded doc-topic incidence:\n", thres_row_norm_doc_topic_df.head())

    # test term-topic incidence
    term_topic_incidence = model.get_term_topic_incidence(doc_ids=doc_ids)
    save_df_to_csv(term_topic_incidence, incidence_save_path, "term_topic_incidence")
    print("first 5 term-topic incidence:\n", term_topic_incidence.head())
