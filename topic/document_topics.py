from top2vec import Top2Vec
import data.files as files

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


def display_wordcloud(model, keyword="tax"):
    topic_words, word_scores, topic_scores, topic_nums = model.search_topics(keywords=[keyword], num_topics=5)
    print(topic_words)
    print(topic_nums)
    for topic in topic_nums:
        model.generate_topic_wordcloud(topic)



if __name__ == '__main__':
    path = "/Users/klara/Downloads"
    pdfs = files.get_files(path=path, file_ending="pdf")
    sentences = []
    for pdf in pdfs:
        sentences.extend(files.extract_text_from_pdf(pdf))

    model = init_document_topics(corpus=sentences)
    print(model.get_topics()[0])
    display_wordcloud(model, keyword="benutzer")