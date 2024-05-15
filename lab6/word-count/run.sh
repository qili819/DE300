







#!/bin/bash

# Running a Docker container with Docker-in-Docker enabled
docker run -v /home/ubuntu/DE300/lab6/word-count:/tmp/wc-demo -it \
           -p 8888:8888 \
           --name word-count-container \
           --privileged \
           -v /var/run/docker.sock:/var/run/docker.sock \
           my-spark-app

