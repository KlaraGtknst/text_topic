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