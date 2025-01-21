import datetime
import json
import logging
from collections import Counter, defaultdict
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
import constants
from utils.logging_utils import *
from utils.os_manipulation import exists_or_create

logger = logging.getLogger(__name__)


class ClusterNamedEntities:

    def __init__(self, client, index: str = constants.DatabaseAddr.DB_NAME, category: str = 'ORG', top_n: int = 50,
                 n_clusters: int = 5, output_file: str = "",
                 es_request_limit: int = 10000, on_server: bool = True):
        """
        Initialize the Named Entity Clustering class.
        :param client: Elasticsearch client, already connected to the server
        :param index: Name of the Elasticsearch index
        :param category: String representing the named entity category to cluster
        :param top_n: Number of top named entities to consider for clustering;
            top in terms of frequency of occurrence across documents
        :param n_clusters: Number of clusters to form
        :param output_file: The path to save the clustering results including the file name and extension.
            If not provided, the results are saved in the server's save path with a default name.
        :param on_server: Boolean indicating whether the code is running on the server or locally
        :param es_request_limit: Number of documents to fetch in each Elasticsearch request at a time.
        """
        self.client = client
        self.index = index
        self.category = category
        self.top_n = top_n
        self.n_clusters = n_clusters
        self.output_file = output_file
        self.es_request_limit = es_request_limit
        init_debug_config(log_filename='cluster_named_entities_', on_server=on_server)

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

        logging.info(f"Fetched {len(named_entities)} named entities for category: {self.category}")
        # Clear the scroll context to free resources on the server
        self.client.clear_scroll(scroll_id=scroll_id)
        return named_entities, doc_map

    def compute_embeddings(self, entities: list):
        """
        Encode entities using SBERT model.
        :param entities: List of named entities to encode
        :return: List of encoded embeddings
        """
        model = SentenceTransformer('sentence-transformers/msmarco-MiniLM-L-12-v3')
        embeddings = model.encode(entities, convert_to_tensor=False)
        logging.info(f"Computed embeddings for {len(entities)} named entities.")
        return embeddings

    def get_top_n_entities(self, named_entities: list):
        """
        Find the top-N most frequent named entities.
        :param named_entities: List of named entities
        :return: List of top-N named entities
        """
        counter = Counter(named_entities)
        top_entities = [entity for entity, _ in counter.most_common(self.top_n)]
        logging.info(f"Top {self.top_n} named entities: {top_entities}")
        return top_entities

    def calculate_similarity_matrix(self, embeddings: list):
        """
        Compute a symmetric similarity matrix for given embeddings.
        :param embeddings: List of embeddings
        :return: Symmetric similarity matrix as array
        """
        return cosine_similarity(embeddings)  # pairwise cosine similarity

    def cluster_named_entities(self, similarity_matrix):
        """
        Perform k-means clustering on the similarity matrix.
        :param similarity_matrix: Symmetric similarity matrix as array obtained from the
            calculate_similarity_matrix(.) method
        :return: List of cluster assignments for each named entity
        """
        kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        clusters = kmeans.fit_predict(similarity_matrix)
        logging.info(f"Performed clustering with {self.n_clusters} clusters.")
        return clusters

    def save_results(self, clusters: list, top_n_entities: list, top_n_doc_maps: dict, embeddings: list):
        """
        Save clustering results, embedding and entity-document mapping of the top N named entities to a JSON file.
        :param clusters: List of cluster assignments for each named entity
        :param top_n_entities: List of top-N named entities
        :param top_n_doc_maps: Mapping of top-N named entities to documents
            (i.e. for each entity in top-N all documents that contain it)
        :param embeddings: List of embeddings for top-N named entities
        :return: None
        """
        result = {
            "category": self.category,
            "clusters": {entity: int(cluster) for entity, cluster in zip(top_n_entities, clusters)},
            "top_n_embeddings": {entity: emb.tolist() for entity, emb in zip(top_n_entities, embeddings)},
            "top_n_entity_document_mapping": top_n_doc_maps,
            # "entity_document_mapping": doc_map    # Uncomment to save the mapping for all entities; may be large!
        }

        output_file = (constants.Paths.SERVER_PLOTS_SAVE_PATH.value + "/cluster_NER/" +
                       f"cluster_NE_results_{self.category}_{datetime.datetime.now().strftime('%x').replace('/', '_')}.json") \
            if self.output_file == "" else self.output_file

        exists_or_create(path=constants.Paths.SERVER_PLOTS_SAVE_PATH.value + "/cluster_NER/")
        with open(output_file, "w") as f:
            json.dump(result, f, indent=4)
        logging.info(f"Results saved to {output_file}")

    def process_category(self, category=""):
        """
        Perform named entity clustering for a specified category with large datasets.
        :param category: String representing the named entity category to cluster;
            if not provided, the default category is used
        """
        if category != "":
            self.category = category
        logging.info(f"Processing category: {self.category}")

        # Step 1: Fetch Named Entities with Scroll API
        named_entities, doc_map = self.fetch_named_entities_with_scroll()
        logging.info(f"Fetched {len(named_entities)} named entities for category: {self.category}")
        if len(named_entities) == 0:
            logging.warning(f"No named entities found for category: {self.category}")
            return

        # Step 2: Compute SBERT Embeddings
        top_n_entities = self.get_top_n_entities(named_entities=named_entities)
        top_n_doc_maps = {entity: doc_map[entity] for entity in top_n_entities}
        embeddings = self.compute_embeddings(entities=top_n_entities)
        logging.info(f"Computed embeddings for {len(top_n_entities)} named entities.")

        # Step 3: Compute Similarity Matrix
        similarity_matrix = self.calculate_similarity_matrix(embeddings=embeddings)
        logging.info("Computed similarity matrix.")

        # Step 4: Perform Clustering
        clusters = self.cluster_named_entities(similarity_matrix=similarity_matrix)
        logging.info(f"Performed clustering with {self.n_clusters} clusters.")

        # Step 5: Save Results
        self.save_results(clusters=clusters, top_n_entities=top_n_entities, doc_map=doc_map,
                          top_n_doc_maps=top_n_doc_maps, embeddings=embeddings)
        logging.info(f"Saved clustering results for category: {self.category}")
