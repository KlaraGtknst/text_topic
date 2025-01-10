# import spacy
# import rel
# # https://rel.readthedocs.io/en/latest/tutorials/e2e_entity_linking/
# # https://github.com/informagi/REL
# from rel.mention_detection import MentionDetection
# from rel.ner import NER
# from rel.entity_disambiguation import EntityDisambiguation
# from elasticsearch import Elasticsearch
#
# # Connect to ElasticSearch
# es = Elasticsearch(["http://localhost:9200"])
#
# def fetch_documents(index, doc_count=10):
#     """Fetch documents from ElasticSearch."""
#     query = {"size": doc_count, "query": {"match_all": {}}}
#     response = es.search(index=index, body=query)
#     return [doc["_source"] for doc in response["hits"]["hits"]]
#
# def process_named_entities(doc):
#     """Link named entities in a document and normalize them."""
#     for entity_type, entities in doc.get("named_entities", {}).items():
#         for i, entity in enumerate(entities):
#             linked_entity = link_entities(entity)
#             if linked_entity:
#                 # Replace entity with its linked canonical version
#                 entities[i] = linked_entity[0][0]  # Replace with the best match
#     return doc
#
# def normalize_documents(index):
#     """Fetch, normalize, and update documents."""
#     documents = fetch_documents(index)
#     for doc in documents:
#         normalized_doc = process_named_entities(doc)
#         print(normalized_doc)  # Replace this with an update to your database
#
# # Run the normalization
# normalize_documents("your_index_name")
#
#
# # Initialize REL components
# mention_detection = MentionDetection(base_url="path/to/rel", wiki_version="wiki_2019")
# ner_model = NER(base_url="path/to/rel", wiki_version="wiki_2019")
# ed_model = EntityDisambiguation(base_url="path/to/rel", wiki_version="wiki_2019")
#
# def link_entities(text):
#     """Link entities in the text using REL."""
#     mentions = mention_detection.find_mentions(text)
#     entities = ed_model.predict(mentions)
#     return entities
#
#
