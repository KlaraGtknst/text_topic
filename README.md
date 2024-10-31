# text_topic
Pipeline to embed .txt documents and cluster them according to their topic.

## Docker Container
To build the docker container, run the following command in the root directory:
```bash
 docker compose up
```
Hence, the elastic search database will be available at `http://localhost:9200`.

After that, you can run the following file to index the documents:
```init_elasticsearch.py
```