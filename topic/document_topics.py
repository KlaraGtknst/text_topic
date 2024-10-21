from top2vec import Top2Vec

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