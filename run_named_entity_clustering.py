from NER.clustering_NE import *
import tqdm


if __name__ == '__main__':
    date = datetime.datetime.now().strftime('%x').replace('/', '_')
    print('File was run at: ', date)
    client = Elasticsearch(constants.CLIENT_ADDR, request_timeout=60)
    top_n = 50
    clusterNamedEntities = ClusterNamedEntities(client=client, index=constants.DB_NAME, top_n=top_n,
                                                n_clusters=top_n // 10)

    # Fetch the index mapping
    mapping = client.indices.get_mapping(index=constants.DB_NAME)
    named_entities_mapping = mapping[constants.DB_NAME]["mappings"]["properties"]["named_entities"]["properties"]

    # Extract the keys (categories)
    categories = list(named_entities_mapping.keys())
    print("All categories of the nested field 'named_entities': ", categories)

    for i in tqdm.tqdm(range(len(categories)), desc='Obtaining named entities clustering for each category'):
        category = categories[i]
        print('--------------------------')
        clusterNamedEntities.process_category(category=category)

    # print('--------------------------')
    # clusterNamedEntities.process_category(category="ORG")
    # print('--------------------------')
    # clusterNamedEntities.process_category(category="PERSON")
