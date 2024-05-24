docker run -v /home/ubuntu/DE300/hw3:/tmp/hw3 -it \
           -p 8888:8888 \
           --name hw3-spark-container \
           --privileged \
           -v /var/run/docker.sock:/var/run/docker.sock \
           spark-app-hw3
