import datetime
import json

import numpy as np
import pandas as pd
import tqdm
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure
from scipy.special import softmax
from top2vec import Top2Vec
from wordcloud import WordCloud

import data.files as files
from data.files import load_sentences_from_file
from utils.os_manipulation import exists_or_create


class TopicModel:

    def __init__(self, documents: list):
        self.model = None
        self.documents = documents
        # pretrained models: universal-sentence-encoder, sentence-transformers
        # model trains own model: doc2vec
        if documents is not None:
            self.create_model()

    def create_model(self):
        self.model = Top2Vec(documents=self.documents,
                             document_ids=list(range(len(self.documents))),
                             # Universal Sentence Encoder multilingual
                             # https://www.sbert.net/docs/sentence_transformer/pretrained_models.html, 20.11.2024
                             embedding_model='distiluse-base-multilingual-cased',
                             speed='fast-learn',
                             workers=8,
                             min_count=10)  # TODO: increase when bigger dataset

    def save_model(self, path: str = "models/"):
        """
        Save the model to a file.
        The name of the file is 'topic_model' and the current date.
        :param path: Path to the file
        :return: -
        """
        exists_or_create(path=path)
        self.model.save(path + "topic_model_" + datetime.datetime.now().strftime('%x').replace('/', '_'))

    def load_model(self, path: str = "models/", filename: str = "topic_model"):
        """
        Load the model from a file.
        :param path: Path to the file
        :return: -
        """
        self.model = Top2Vec.load(path + filename)

    def get_num_topics(self):
        """
        This function returns the number of topics found in TopicModel.
        :return: Number of topics
        """
        return self.model.get_num_topics()

    def get_closest_topics(self, word: str, num_topics: int):
        """
        Semantic search of topics using text query.

        These are the topics closest to the vector. Topics are ordered by
        proximity to the vector. Successive topics in the list are less
        semantically similar to the vector.

        :param word: Text query
        :param num_topics: Number of topics to return
        :return: array of most similar topics described by 50 most similar words; shape: (num topics, 50)
        """
        return self.model.query_topics(word, num_topics=num_topics)

    def get_closest_documents(self, word: str, num_docs: int):
        """
        Semantic search of documents using text query.
        :param word: Text query
        :param num_docs: Number of documents to return
        :return: Documents, their score and ID in descending order of similarity
        """
        return self.model.query_documents(word, num_docs=num_docs, return_documents=True)

    def get_wordcloud_of_similar_topics(self, num_topics: int, word: str = None):
        """
        This function creates a wordcloud of the topics most similar to the word.
        :param num_topics: Number of topics to return
        :param word: Text query
        :return: -
        """
        topic_words, word_scores, topic_scores, topic_nums = self.model.search_topics(keywords=[word],
                                                                                      num_topics=num_topics)
        fig = Figure(figsize=(10, 4))
        canvas = FigureCanvasAgg(fig)
        for topic_num in topic_nums:
            self.model._validate_topic_num(topic_num, False)
            word_score_dict = dict(zip(self.model.topic_words[topic_num],
                                       softmax(self.model.topic_word_scores[topic_num])))
            wordcloud = WordCloud(width=1000, height=1000, background_color='white').generate_from_frequencies(
                word_score_dict)
            plt.axis('off')
            plt.title('Topic ' + str(topic_num))
            plt.imshow(wordcloud)
            canvas.draw()
            plt.show()

    def get_doc_topics(self, doc_ids: list, num_topics: int = 1):
        """
        This function returns the topics of documents.
        :param doc_ids: List of document ids
        :param num_topics: Number of topics to return
        :return: Topics of documents
        """
        topic_nums, topic_score, topics_words, word_scores = self.model.get_documents_topics(doc_ids=doc_ids,
                                                                                             num_topics=num_topics)
        print("obtained document topics")
        return topic_nums, topic_score, topics_words, word_scores

    def get_document_topic_incidence(self, doc_ids: list):
        """
        This function returns the incidence of topics in documents.
        :param doc_ids: List of document ids
        :return: Incidence of topics in documents
        """
        # default number of topics returned by model is 1
        topic_nums, topic_score, topics_words, word_scores = self.get_doc_topics(doc_ids=doc_ids, num_topics=10)

        # create dataframe with document-topic incidence
        # use num_docs instead of doc_ids, bc here we want to index return object not topic model object
        doc_topic_columns = {topic_num: [0 if topic_num not in topic_nums[doc_id] else \
                                             topic_score[doc_id][np.where(topic_nums[doc_id] == topic_num)[0][0]] \
                                         for doc_id in range(len(topic_score))] for topic_num in
                             range(self.model.get_num_topics())}

        # real values are topic scores in [0, 1]
        document_topic_incidence = pd.DataFrame(doc_topic_columns)  # automatic index == document id in TopicModel

        return document_topic_incidence

    def get_term_topic_incidence(self, doc_ids: list, save_path_topic_words: str = None):
        """
        This function returns the incidence of terms in topics.
        :param doc_ids: List of document ids
        :param save_path_topic_words: Path to save the topics words including file name and json ending; if None, no saving
        :return: Incidence of terms in topics
        """
        num_topics = self.get_num_topics()
        topic_nums, topic_score, topics_words, word_scores = self.get_doc_topics(doc_ids=doc_ids, num_topics=10)
        num_docs = len(topic_score)

        # create term-topic incidence dataframe
        # use num_docs instead of doc_ids, bc here we want to index return object not topic model object
        topic_index_per_doc = {topic_num: [-1 if topic_num not in topic_nums[doc_id]
                                           else np.where(topic_nums[doc_id] == topic_num)[0][0]
                                           for doc_id in range(num_docs)] for topic_num in range(num_topics)}

        terms_per_topic = {topic_num: {term for doc_id in range(num_docs)
                                       if topic_index_per_doc[topic_num][doc_id] != -1
                                       for term in topics_words[doc_id][topic_index_per_doc[topic_num][doc_id]]}
                           for topic_num in range(num_topics)}

        if save_path_topic_words:
            exists_or_create(path=''.join(save_path_topic_words.split('/')[:-1]))
            if not save_path_topic_words.endswith('.json'):
                save_path_topic_words += '.json'
            # Save to JSON file
            with open(save_path_topic_words, "w") as f:
                json.dump(terms_per_topic, f, indent=4)

        print("obtained terms per topic")
        term_topic_columns = {term: [term in terms_per_topic[topic_num] for topic_num in range(num_topics)] for term in
                              self.model.vocab}

        print("obtained term_topic_columns")

        # values are binary: 1 if term is in topic, 0 otherwise
        term_topic_incidence = pd.DataFrame(term_topic_columns)  # automatic index == term id in TopicModel
        return term_topic_incidence

    def row_normalize_df(self, df):
        """
        This function normalizes the rows of a dataframe.
        :param df: Dataframe to normalize
        :return: Normalized dataframe
        """
        return df.div(df.sum(axis=1), axis=0)

    def get_density_doc_topic_threshold(self, normalized_doc_topic_incidence, threshold: float):
        """
        This function returns the density of the document-topic incidence matrix.
        :param normalized_doc_topic_incidence: Normalized document-topic incidence matrix
        :param threshold: Threshold for weights in the matrix to be considered as relevant
        :return: Density of the document-topic incidence matrix (proportion of weights above threshold)
        """
        return np.mean(normalized_doc_topic_incidence > threshold)

    def display_density_doc_topic_threshold(self, normalized_doc_topic_incidence, save_path: str = None,
                                            opt_density: float = 0.1):
        """
        This function displays the density of the document-topic incidence matrix for different thresholds.
        :param normalized_doc_topic_incidence: Normalized document-topic incidence matrix
        :return: Density of the document-topic incidence matrix (proportion of weights above threshold)
        """
        plt.figure()
        thresholds = np.linspace(0, 1, 100)
        densities = [self.get_density_doc_topic_threshold(normalized_doc_topic_incidence, threshold) for threshold in
                     thresholds]
        plt.fill_between(thresholds, densities, color='skyblue', alpha=0.6, label='Density')
        if np.where(np.array(densities) - opt_density > 0)[0].size == 0:  # if only negative values, array is empty
            opt_threshold = np.max(densities)  # keep all values
        else:
            opt_threshold = thresholds[np.where(np.array(densities) - opt_density > 0)[0][-1]]

        plt.axvline(x=opt_threshold, color='purple', linestyle='--',
                    label=f'Optimal threshold = {np.round(opt_threshold, decimals=2)} '
                          f'for density = {opt_density}')
        plt.xlabel('Threshold')
        plt.ylabel('Density')
        title = 'Density of document-topic incidence matrix'
        plt.legend()
        plt.title(title)
        if save_path:
            date = datetime.datetime.now().strftime('%x').replace('/', '_')
            exists_or_create(path=save_path)
            plt.savefig(save_path + title + date + '.svg', format='svg')
        plt.show()
        return densities, thresholds, opt_threshold

    def determine_threshold_doc_topic_threshold(self, doc_topic_incidence, opt_density: float = 0.1,
                                                save_path: str = None):
        """
        This function displays the density of the document-topic incidence matrix for different thresholds.
        :param doc_topic_incidence: Document-topic incidence matrix
        :param save_path: Path to save the plot; if None, no saving
        :param opt_density: Optimal density of the document-topic incidence matrix
        :return: Optimal density of the document-topic incidence matrix (proportion of weights above threshold),
                Row normalized document-topic incidence matrix
        """
        normalized_doc_topic_incidence = self.row_normalize_df(doc_topic_incidence)
        densities, thresholds, opt_threshold = self.display_density_doc_topic_threshold(normalized_doc_topic_incidence,
                                                                                        save_path=save_path,
                                                                                        opt_density=opt_density)
        return opt_threshold, normalized_doc_topic_incidence

    def apply_threshold_doc_topic_incidence(self, doc_topic_incidence, threshold: float = None):
        """
        This function applies a threshold to the document-topic incidence matrix.
        :param doc_topic_incidence: Document-topic incidence matrix;
                if threshold is None, row normalized matrix is calculated;
                if threshold is provided, input matrix is assumed to be row normalized
        :param threshold: (Optional) Threshold for weights in the matrix to be considered as relevant;
                if None, optimal threshold is determined
        :return: Document-topic incidence matrix with threshold applied
        """
        if threshold is None:
            # determine optimal threshold
            # overwrites doc_topic_incidence with row normalized version
            threshold, doc_topic_incidence = self.determine_threshold_doc_topic_threshold(doc_topic_incidence)
        return doc_topic_incidence.map(lambda x: 1 if x > threshold else 0)

    @classmethod
    def TopicModel(cls, documents):
        pass


if __name__ == '__main__':
    path = "/Users/klara/Documents/uni/"
    dataset_path = "../dataset/"
    model_path = '../models/'
    incidence_save_path = "../results/incidences/"
    plot_save_path = "../results/plots/"

    # texts
    if dataset_path:
        sentences = load_sentences_from_file(dataset_path)
        sentences = sentences.split('NEWFILE')
    else:
        pdfs = files.get_files(path=path)
        sentences = []
        for i in tqdm.tqdm(range(len(pdfs)), desc='Extracting text from pdfs'):
            pdf = pdfs[i]
            sentence = files.extract_text_from_pdf(pdf)
            if type(sentence) != str:
                sentence = str(sentence)
            sentences.extend([sentence])
        #save_sentences_to_file(sentences, dataset_path)

    if model_path:
        model = TopicModel(None)
        model.load_model(path=model_path)

    else:
        model = TopicModel(documents=sentences)
        model.save_model(path=model_path)  # unique name with date

    # document-topic incidence
    start = 0
    duration = len(sentences)
    doc_ids = list(range(start, start + len(sentences[start:start + duration]) - 1))
    #print(doc_ids)

    doc_topic_incidence = model.get_document_topic_incidence(doc_ids=doc_ids)
    #save_df_to_csv(doc_topic_incidence, incidence_save_path, "doc_topic_incidence")
    print("first 5doc-topic incidence:\n", doc_topic_incidence.head())

    # determine optimal threshold for document-topic incidence
    threshold, row_norm_doc_topic_df = model.determine_threshold_doc_topic_threshold(doc_topic_incidence,
                                                                                     opt_density=0.1,
                                                                                     save_path=plot_save_path)
    print("optimal threshold: ", threshold)
    thres_row_norm_doc_topic_df = model.apply_threshold_doc_topic_incidence(row_norm_doc_topic_df, threshold=threshold)
    #save_df_to_csv(thres_row_norm_doc_topic_df, incidence_save_path, "thres_row_norm_doc_topic_incidence")
    print("first 5 thresholded doc-topic incidence:\n", thres_row_norm_doc_topic_df.head())

    # test term-topic incidence
    term_topic_incidence = model.get_term_topic_incidence(doc_ids=doc_ids)
    # save_df_to_csv(term_topic_incidence, incidence_save_path, "term_topic_incidence")
    print("first 5 term-topic incidence:\n", term_topic_incidence.head())
