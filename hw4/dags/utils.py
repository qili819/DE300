from pyspark.sql import SparkSession
from scripts.eda_spark import DataCleaner
from scripts.fe_spark import FeatureEngineeringSpark
from scripts.train_model_spark import TrainModelSpark

def load_data():
    from scripts.load_data import main
    main()

def eda_standard():
    from scripts.eda_standard import main
    main()

def eda_spark():
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("EDA Spark") \
        .getOrCreate()

    # Read the data
    data = spark.read.csv('/tmp/data.csv', header=True, inferSchema=True)

    # Apply the DataCleaner transformer
    cleaner = DataCleaner()
    cleaned_data = cleaner.transform(data)

    # Optionally, you can save the cleaned data to a file for further processing
    cleaned_data.write.csv('/tmp/cleaned_data.csv', header=True, mode='overwrite')

    # Stop the Spark session
    spark.stop()

def fe_standard():
    from scripts.fe_standard import main
    main()

def fe_spark():
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("Feature Engineering Spark") \
        .getOrCreate()

    # Initialize the FeatureEngineeringSpark class with Spark session
    fe_spark_instance = FeatureEngineeringSpark(spark)
    
    # Transform the data
    fe_spark_instance.transform_data('/tmp/cleaned_data.csv', '/tmp/fe_spark_data.csv')
    
    # Stop the Spark session
    spark.stop()

def train_model_standard():
    from scripts.train_model_standard import main
    main()

def train_model_spark():
    # Initialize Spark session
    spark = SparkSession.builder \
        .appName("Train Model Spark") \
        .getOrCreate()

    # Initialize the TrainModelSpark class with Spark session
    train_model_instance = TrainModelSpark(spark)
    
    # Train and evaluate the model
    train_model_instance.train_spark('/tmp/fe_spark_data.csv')
    
    # Stop the Spark session
    spark.stop()

def select_best_model():
    from scripts.select_best_model import main
    main()
