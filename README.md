# text_topic
This repository implements a pipeline to store various data of files from a large unstructured dataset. 
These fields are used for topic modeling (wordclouds, based on low-dimensional versions of embedding vectors, Named Entity Clustering and document-topic incidences). 
The information is aggregated and visualised using FCA. 

## Docker Container
To build the docker container, run the following command in the root directory of the project on server:
```bash
 docker compose up -d
```
Hence, the elastic search database will be available at `http://localhost:9200`.
**-d flag** stands for detached mode. 
When you include this flag, 
Docker Compose runs the containers in the background and detaches the process from your terminal.

## Elasticsearch Database
Before using the database on the watzmann server, you need to create a tunnel to the server
(i.e. run it locally on your machine).:
```bash
ssh -L 9200:localhost:9200 watzmann
```
This is ssh port forwarding: 
port forwarding or port mapping is an application of network address translation (NAT) that redirects a communication 
request from one address and port number combination to another. (https://en.wikipedia.org/wiki/Port_forwarding, 22.12.2024)

- first 9200: This is the port on your local machine (your client machine) where the SSH tunnel will listen for connections.
- localhost: This is the destination host from the perspective of the SSH server (in this case, the watzmann server).
- second 9200: This is the port on the destination host (from the server's perspective) to which traffic should be forwarded. 
- watzmann: This is the remote host (the SSH server) you are connecting to.

When you run this command:
1. You establish an SSH connection to the remote host watzmann.
2. A tunnel is created between your local machine and watzmann.
3. Any connection made to localhost:9200 on your local machine will be securely forwarded to localhost:9200 on the watzmann machine.

## Start the Pipeline of filling the index
We have chosen to split the pipeline into two parts, because the text related fields are computed on a different server 
which has more graphical computation power.
You can see the workflow in the following image:

![es_workflow.svg](doc/es_workflow.svg)

You can run the following file to initialize the index (i.e. run on the watzmann server):
```bash
python3 main.py
```
Hence, the index is created, but no documents are indexed yet.
To index the documents and insert their metadata, run the following command (i.e. run on the watzmann server):
```bash
python3 insert_metadata.py
```
![text_related_workflow.svg](doc/text_related_workflow.svg)

To insert the text related fields of the documents, run the following command (i.e. run on the pumbaa server):
```bash 
python3 insert_text_related_fields.py
```
As you can see in the image, the text related fields compromise of three fields.
The first field is the text of a file either obtained directly, via a PdfReader or via an ImageCaptioner.
The second field is the embeddings of the text, which are computed by the `sentence-transformers` library 
([SBERT](https://huggingface.co/sentence-transformers/msmarco-MiniLM-L-12-v3)).
The third field is a nested structure containing Named Entities of the text, 
obtained using the small english pipeline `en_core_web_sm` of the [spaCy](https://spacy.io/models) library.


## Obtain incidences
With reference to ["The Geometric Structure of Topic Models", Johannes Hirth and Tom Hanika (2024)](https://arxiv.org/abs/2403.03607),
we obtain the document-topic and topic-word incidences.

![doc_topic_inc_fca_hirth_hanika.svg](doc/doc_topic_inc_fca_hirth_hanika.svg)

You can run the following command to obtain the incidences (i.e. run on the watzmann server):
```bash
python3 create_fca_incidences.py
```

## Incidences to Context
Once the incidences are produced, you want to convert them to the FIMI format:
```bash
phyton3 run_topic_fca.py
```
This call will also create document-topic incidences per directory which will be used later to compute a directory-topic context.

The FIMI files can be used to compute their intents via PCBO (FCALGS).
This algorithm is implemented in the `fcalgs` package.
You need to install the package first (i.e. run on the watzmann server):
```bash
wget https://sourceforge.net/projects/fcalgs/files/pcbo/amai/pcbo-amai.zip
unzip pcbo-amai.zip
cd pcbo-amai
make
``` 
(website worked on 09.01.2025)

After that, you can run the following command to compute the intents (in the `pcbo-amai` directory):
```bash
./pcbo -P4 /file/to/fimi/file.fimi /name/of/output/file.fimi
```
# Topic Modeling Strategies

## Wordclouds & 2D scatter of documents coloured by their directory
To run both strategies above (and more, you might have to comment functions you don't need),
run the following command:
```bash
python3 visualizations.py
```
![wordcloud_Military.svg](doc/wordcloud_Military.svg)
![TSNE_scatter_documents_dir_2d_01_22_25.png](doc/TSNE_scatter_documents_dir_2d_01_22_25.png)
## Named Entity Clustering
Similar to ["Clustering Prominent Named Entities in Topic-Specific Text Corpora", A. Alsudais and H. Tchalian (2019)](https://arxiv.org/pdf/1807.10800),
we cluster named entities of different categories across the text extracted from the files of the dataset.
To cluster the named entities, you can run the following command (i.e. run on the watzmann server):
```bash
python3 run_named_entity_clustering.py
```
The workflow is displayed in the following image:
![text_related_workflow.svg](doc/NE_Clustering.svg)
The results vary in quality strongly depending on the NER and text quality.
An example of the clustering (of dataset [EYNTKE](https://archive.org/details/ETYNTKE)) is shown in the following image.
Different language families are well separated, but the topological structure forms no clear clusters.
![named_entity_clusters_LANGUAGE_PCA_01_22_25.svg](doc/named_entity_clusters_LANGUAGE_PCA_01_22_25.svg)

## Supplementary Information
Ensure that the transformers library version is == 4.48.0.
As of 16.01.2025, using transformers == 4.45.0 will result in an error (on the server), since the processor used for the 
image captioner could not be loaded.

Make sure that the pretrained models are downloaded:
```python -m spacy download en_core_web_sm```
