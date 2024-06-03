from pyspark.sql import SparkSession
from pyspark.ml.feature import ChiSqSelector, Bucketizer, VectorAssembler
from pyspark.sql.functions import col

class FeatureEngineeringSpark:
    def __init__(self, spark: SparkSession):
        self.spark = spark

    def feature_transform_data(self, input_path: str, output_path: str):
        # Read the cleaned data
        data = self.spark.read.csv(input_path, header=True, inferSchema=True)

        # Assemble features into a single vector
        feature_columns = [col for col in data.columns if col != 'num']
        assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")
        assembled_data = assembler.transform(data)

        # Feature Selection using Chi-Squared
        selector = ChiSqSelector(numTopFeatures=10, featuresCol="features", outputCol="selectedFeatures", labelCol="num")
        selected_data = selector.fit(assembled_data).transform(assembled_data)

        # Binning a continuous variable (example: age)
        splits = [-float("inf"), 30, 40, 50, 60, float("inf")]
        bucketizer = Bucketizer(splits=splits, inputCol="age", outputCol="age_binned")
        binned_data = bucketizer.transform(selected_data)

        # Drop original features and keep selected and binned features
        final_columns = [col for col in selected_data.columns if col not in feature_columns] + ['age_binned']
        final_data = binned_data.select(final_columns)

        # Save the transformed data
        final_data.write.csv(output_path, header=True, mode='overwrite')

