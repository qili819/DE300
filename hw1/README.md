Instruction for deploying:

1) Run the following block to clear docker containers and networks:
docker stop postgres-container
docker stop etl-container 
docker rm postgres-container 
docker rm etl-container 
docker network rm etl-database

2) Run the run.sh file to create initiate all required containers(etl-container and postgres-container) and enter the shell of etl-container.
3) Start the jupyter notebook by 'jupyter notebook --ip=0.0.0.0'.
4) Run all cells in the jupyter notebook de300hw1.ipynb
