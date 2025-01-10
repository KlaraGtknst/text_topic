import datetime

from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import json
from collections import Counter, defaultdict

import constants


class ClusterNamedEntities:

    def __init__(self, client, index=constants.DB_NAME, category='ORG', top_n=50, n_clusters=5, output_file="",
                 es_request_limit=10000):
        self.client = client
        self.index = index
        self.category = category
        self.top_n = top_n
        self.n_clusters = n_clusters
        self.output_file = output_file
        self.es_request_limit = es_request_limit

    def fetch_named_entities_with_scroll(self):
        """
        Fetch named entities of the specified category using the scroll API for large datasets.
        """
        named_entities = []
        doc_map = defaultdict(list)  # Map named entities to documents

        # Initialize the scroll request
        query = {
            "size": self.es_request_limit,
            "_source": [f"named_entities.{self.category}"],
            "query": {
                "nested": {
                    "path": "named_entities",
                    "query": {
                        "exists": {
                            "field": f"named_entities.{self.category}"
                        }
                    }
                }
            }}

        response = self.client.search(index=self.index, body=query, scroll="2m")
        scroll_id = response["_scroll_id"]

        # Process the first batch of results
        while True:
            hits = response["hits"]["hits"]
            if not hits:
                break

            for doc in hits:
                doc_id = doc["_id"]
                entities = doc["_source"].get("named_entities", {}).get(self.category, [])
                named_entities.extend(entities)
                for entity in entities:
                    doc_map[entity].append(doc_id)

            # Fetch the next batch of results
            response = self.client.scroll(scroll_id=scroll_id, scroll="2m")

        # Clear the scroll context to free resources on the server
        self.client.clear_scroll(scroll_id=scroll_id)
        return named_entities, doc_map

    def compute_embeddings(self, entities):
        """
        Encode entities using SBERT model.
        """
        model = SentenceTransformer('sentence-transformers/msmarco-MiniLM-L-12-v3')
        embeddings = model.encode(entities, convert_to_tensor=False)
        return embeddings

    def get_top_n_entities(self, named_entities):
        """
        Find the top-N most frequent named entities.
        """
        counter = Counter(named_entities)
        top_entities = [entity for entity, _ in counter.most_common(self.top_n)]
        return top_entities

    def calculate_similarity_matrix(self, embeddings):
        """
        Compute a symmetric similarity matrix for given embeddings.
        """
        return cosine_similarity(embeddings)

    def cluster_named_entities(self, similarity_matrix):
        """
        Perform k-means clustering on the similarity matrix.
        """
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        clusters = kmeans.fit_predict(similarity_matrix)
        return clusters

    def save_results(self, clusters, top_n_entities, doc_map, top_n_doc_maps, embeddings):
        """
        Save clustering results and entity-document mapping to a JSON file.
        """
        result = {
            "category": self.category,
            "clusters": {entity: int(cluster) for entity, cluster in zip(top_n_entities, clusters)},
            "top_n_embeddings": {entity: emb.tolist() for entity, emb in zip(top_n_entities, embeddings)},
            "top_n_entity_document_mapping": top_n_doc_maps,
            # "entity_document_mapping": doc_map    # Uncomment to save the mapping for all entities; may be large!
        }

        output_file = (constants.SERVER_SAVE_PATH + "/cluster_NER/" +
                       f"cluster_NE_results_{self.category}_{datetime.datetime.now().strftime('%x').replace('/', '_')}.json") \
            if self.output_file == "" else self.output_file

        with open(output_file, "w") as f:
            json.dump(result, f, indent=4)
        print(f"Results saved to {output_file}")

    def process_category(self, category=""):
        """
        Perform named entity clustering for a specified category with large datasets.
        """
        if category != "":
            self.category = category
        print(f"Processing category: {self.category}")

        # Step 1: Fetch Named Entities with Scroll API
        named_entities, doc_map = self.fetch_named_entities_with_scroll()
        print(f"Fetched {len(named_entities)} named entities.")
        if len(named_entities) == 0:
            print(f"No named entities found for category: {self.category}")
            return

        # Step 2: Compute SBERT Embeddings
        top_n_entities = self.get_top_n_entities(named_entities=named_entities)
        top_n_doc_maps = {entity: doc_map[entity] for entity in top_n_entities}
        embeddings = self.compute_embeddings(entities=top_n_entities)

        # Step 3: Compute Similarity Matrix
        similarity_matrix = self.calculate_similarity_matrix(embeddings=embeddings)

        # Step 4: Perform Clustering
        clusters = self.cluster_named_entities(similarity_matrix=similarity_matrix)

        # Step 5: Save Results
        self.save_results(clusters=clusters, top_n_entities=top_n_entities, doc_map=doc_map,
                          top_n_doc_maps=top_n_doc_maps, embeddings=embeddings)
