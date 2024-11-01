import glob
from matplotlib import pyplot as plt
from matplotlib.figure import Figure
import io
from top2vec import Top2Vec
from scipy.special import softmax
from wordcloud import WordCloud
from matplotlib.backends.backend_agg import FigureCanvasAgg
from PIL import Image
import data.files as files

class TopicModel():

    def __init__(self, documents: list):
        self.documents = documents
        # pretrained models: universal-sentence-encoder, sentence-transformers
        # model trains own model: doc2vec
        self.model = Top2Vec(documents=self.documents,
                             embedding_model='distiluse-base-multilingual-cased',  # SBERT
                             speed='fast-learn',
                             workers=8,
                             min_count=10) # TODO: increase when bigger dataset

    def get_num_topics(self):
        '''
        This function returns the number of topics found in TopicModel.
        :return: Number of topics
        '''
        return self.model.get_num_topics()

    def get_closest_topics(self, word:str, num_topics:int):
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

    def get_closest_documents(self, word:str, num_docs:int):
        """
        Semantic search of documents using text query.
        :param word: Text query
        :param num_docs: Number of documents to return
        :return: Documents, their score and ID in descending order of similarity
        """
        return self.model.query_documents(word, num_docs=num_docs, return_documents=True)

    def get_wordcloud_of_similar_topics(self, num_topics:int, word:str=None):
        """
        This function creates a wordcloud of the topics most similar to the word.
        :param num_topics: Number of topics to return
        :param word: Text query
        :return: -
        """
        topic_words, word_scores, topic_scores, topic_nums = self.model.search_topics(keywords=[word], num_topics=num_topics)
        fig = Figure(figsize=(10, 4))
        canvas = FigureCanvasAgg(fig)
        for topic_num in topic_nums:
            self.model._validate_topic_num(topic_num, False)
            word_score_dict = dict(zip(self.model.topic_words[topic_num],
                                       softmax(self.model.topic_word_scores[topic_num])))
            wordcloud = WordCloud(width=1000, height=1000, background_color='white').generate_from_frequencies(word_score_dict)
            plt.axis('off')
            plt.title('Topic ' + str(topic_num))
            plt.imshow(wordcloud)
            canvas.draw()
            plt.show()


if __name__ == '__main__':
    path = "/Users/klara/Documents/uni/"

    # texts
    pdfs = files.get_files(path=path, file_ending="pdf")
    sentences = []
    for pdf in pdfs:
        sentences.extend(files.extract_text_from_pdf(pdf))

    model = TopicModel(documents=sentences)
    print('closest topics:', model.get_closest_topics(word='benutzer', num_topics=1)[0])
    model.get_wordcloud_of_similar_topics(num_topics=2, word="benutzer")
