# build the image
docker build -t pyscrape:0.1 .

# run the container in interactive mode with port forwarding
docker run -v "$(pwd)/scraping":/app/scraping \
-p 8888:8888 \
--rm -it pyscrape:0.1 /bin/bash