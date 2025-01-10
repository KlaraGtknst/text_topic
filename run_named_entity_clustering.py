import datetime

import constants
from NER.clustering_NE import *


if __name__ == '__main__':
    date = datetime.datetime.now().strftime('%x').replace('/', '_')
    print('File was run at: ', date)
    client = Elasticsearch(constants.CLIENT_ADDR)
    clusterNamedEntities = ClusterNamedEntities(client=client, index=constants.DB_NAME, top_n=50, n_clusters=5)
    print('--------------------------')
    clusterNamedEntities.process_category(category="ORG")
    print('--------------------------')
    clusterNamedEntities.process_category(category="PERSON")
