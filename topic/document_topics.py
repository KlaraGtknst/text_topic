from top2vec import Top2Vec
import data.files as files
import matplotlib.pyplot as plt

def init_document_topics(corpus, model=None):
    if model is None:
        model = Top2Vec(corpus,
                        embedding_model='distiluse-base-multilingual-cased',    # SBERT
                        speed='fast-learn')
    return model

def get_num_topics(model):
    return model.get_num_topics()

def get_topic_info(model):
    return model.get_topics()


def display_wordcloud(model, keyword="tax", max_num_topics=2):
    topic_words, word_scores, topic_scores, topic_nums = model.search_topics(keywords=[keyword],
                                                                             num_topics=max_num_topics
                                                                             )
    #print(topic_words, word_scores, topic_scores, topic_nums)
    for topic_num in topic_nums:
        try:
            model.generate_topic_wordcloud(topic_num)
            plt.title('Topic ' + str(topic_num))
            plt.show()
        except ValueError:
            print(f"Topic {topic_num} not found in model")





if __name__ == '__main__':
    path = "/Users/klara/Downloads"
    pdfs = files.get_files(path=path, file_ending="pdf")
    sentences = []
    for pdf in pdfs:
        sentences.extend(files.extract_text_from_pdf(pdf))

    model = init_document_topics(corpus=sentences)
    print(model.get_topics()[0])
    display_wordcloud(model, keyword="Deep")