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
                             workers=8)

    def get_num_topics(self):
        return self.model.get_num_topics()

    def get_closest_topics(self, word:str, num_topics:int):
        return self.model.query_topics(word, num_topics=num_topics)

    def get_closest_documents(self, word:str, num_docs:int):
        return self.model.query_documents(word, num_docs=num_docs)

    def get_wordcloud_of_similar_topics(self, num_topics:int, word:str=None):
        topic_words, word_scores, topic_scores, topic_nums = self.model.search_topics(keywords=[word], num_topics=num_topics)
        fig = Figure(figsize=(10, 4))
        canvas = FigureCanvasAgg(fig)
        for topic_num in topic_nums:
            self.model._validate_topic_num(topic_num, False)
            word_score_dict = dict(zip(self.model.topic_words[topic_num],
                                       softmax(self.model.topic_word_scores[topic_num])))
            wordcloud = WordCloud(width=1000, height=1000, background_color='white').generate_from_frequencies(word_score_dict)
            ax = fig.add_subplot(1, num_topics, topic_num+1)
            ax.axis('off')
            ax.set_title('Topic ' + str(topic_num))
            ax.imshow(wordcloud)
        canvas.draw()
        return Image.frombytes('RGB', canvas.get_width_height(), canvas.buffer_rgba())


if __name__ == '__main__':
    path = "/Users/klara/Downloads"
    pdfs = files.get_files(path=path, file_ending="pdf")
    sentences = []
    for pdf in pdfs:
        sentences.extend(files.extract_text_from_pdf(pdf))

    model = TopicModel(documents=sentences)
    print('closest topics:', model.get_closest_topics(word='benutzer', num_topics=1)[0])
    img_data = model.get_wordcloud_of_similar_topics(num_topics=2, word="benutzer")
    imgplot = plt.imshow(img_data)
    plt.show()