# text_topic
Pipeline to embed .txt documents and cluster them according to their topic.

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

## Start the Pipeline
After that, you can run the following file to index the documents (i.e. run on the watzmann server):
```bash
python3 main.py
```

## Incidences to Context
Once the incidences are produced by the main.py file, 
you want to convert them to the FIMI format:
```bash
phyton3 run_topic_fca.py
```

These FIMI files can be used to compute their intents via PCBO (FCALGS).
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


## Supplementary Information
Ensure that the transformers library version is == 4.48.0.
As of 16.01.2025, using transformers == 4.45.0 will result in an error (on the server), since the processor used for the 
image captioner could not be loaded.

Make sure that the pretrained models are downloaded:
```python -m spacy download en_core_web_sm```
