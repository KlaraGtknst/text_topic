import tqdm
from NER.clustering_NE import *
from database.init_elasticsearch import ESDatabase

if __name__ == '__main__':
    on_server = True
    # init_debug_config(log_filename='run_named_entity_clustering_', on_server=on_server)
    init_debug_config(log_filename='run_kmeans_elbow_', on_server=on_server)

    es_db = ESDatabase()
    client = es_db.get_es_client()
    top_n = 50
    clusterNamedEntities = ClusterNamedEntities(client=client, index=constants.DatabaseAddr.DB_NAME.value, top_n=top_n,
                                                n_clusters=top_n // 10)

    # Fetch the index mapping
    mapping = client.indices.get_mapping(index=constants.DatabaseAddr.DB_NAME.value)
    named_entities_mapping = mapping[constants.DatabaseAddr.DB_NAME.value]["mappings"]["properties"]["named_entities"]["properties"]

    # Extract the keys (categories)
    categories = list(named_entities_mapping.keys())
    logging.info(f'All categories of the nested field "named_entities": {categories}')

    for i in tqdm.tqdm(range(len(categories)), desc='Obtaining named entities clustering for each category'):
        category = categories[i]
        logging.info(f'Processing category: {category}')
        clusterNamedEntities.process_category(category=category)
