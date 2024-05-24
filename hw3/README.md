Files Explanation:
Dockerfile: Dockerfile for building image
run.sh: file to create docker container
data: folder with raw data used during local implementation
main.py: Local implementation script
hw3nb.ipynb: notebook to execute with Jupyternb
run-py-spark.sh: file to run pyspark code locally
bootstrap.sh: bootstrap file for EMR
hw3script.py: script for EMR implementation


Local implementation:
1. Build docker image using: docker build -t spark-app-hw3 .
2. Bash run.sh to create docker container
3. Enter the container and activate the virtual environment
4. Bash run-py-spark.sh to execute main.py
5. Output: print statements that outline the type of model chosen (Random Forest or Logistic Regression), Parameters and AUC

EMR implementation:
use bootstrap.sh and hw3script to complete implementation on EMR
