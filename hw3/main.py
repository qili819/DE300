#main script for hw3
from pyspark.sql import SparkSession
from pyspark.sql.types import IntegerType, FloatType, LongType, StringType, DoubleType
from pyspark.sql.dataframe import DataFrame
from pyspark.sql.functions import col, when
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline, Transformer
from pyspark.ml.feature import StringIndexer, VectorAssembler, Imputer
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.param.shared import HasInputCol, HasOutputCol
import pyspark.sql.functions as F
from itertools import combinations
import os

# Check Python path
import sys
sys.executable

DATA_FOLDER = "data"
NUMBER_OF_FOLDS = 5
SPLIT_SEED = 7576
TRAIN_TEST_SPLIT = 0.9

def read_data(spark: SparkSession) -> DataFrame:
    """
    read data; since the data has the header we let spark guess the schema
    """
    
    # Read the hd CSV data into a DataFrame
    hd_data = spark.read \
        .format("csv") \
        .option("header", "true") \
        .option("inferSchema", "true") \
        .load(os.path.join(DATA_FOLDER,"*.csv"))

    return hd_data


###### Transformer Class ######
class DataCleaner(Transformer):
    def _transform(self, dataset: DataFrame) -> DataFrame:
        # Select the necessary columns
        dt = dataset.select('age', 'sex', 'painloc', 'painexer', 'cp', 'trestbps', 'fbs', 
                            'prop', 'nitr', 'pro', 'diuretic', 'thaldur', 'thalach', 'exang', 
                            'oldpeak', 'slope', 'num')

        # Convert all columns to numeric, handling errors by coercion
        for column in dt.columns:
            dt = dt.withColumn(column, col(column).cast('float'))

        dt = dt.withColumn('num', F.when(F.col('num') == 0, 0).otherwise(1))

        # a. Filter painloc and painexer to be either 1 or 0
        dt = dt.filter((col('painloc') >= 0) & (col('painexer') >= 0))
        dt = dt.withColumn('painloc', when(col('painloc') > 1, 1).otherwise(col('painloc')))
        dt = dt.withColumn('painexer', when(col('painexer') > 1, 1).otherwise(col('painexer')))

        # b. Replace trestbps < 100 with 0
        dt = dt.withColumn('trestbps', when(col('trestbps') < 100, 0).otherwise(col('trestbps')))

        # c. Replace oldpeak < 0 with 0 and oldpeak > 4 with 4
        dt = dt.withColumn('oldpeak', when(col('oldpeak') < 0, 0).when(col('oldpeak') > 4, 4).otherwise(col('oldpeak')))

        # d. Remove rows with missing thaldur and thalach
        dt = dt.filter((col('thaldur') != -9) & (col('thalach') != -9))

        # e. Remove rows with missing fbs, prop, nitr, pro, diuretic, adjusting prop to be 0 or 1
        dt = dt.filter((col('fbs') != -9) & (col('prop') != -9) & (col('nitr') != -9) &
                       (col('pro') != -9) & (col('diuretic') != -9))
        dt = dt.withColumn('prop', when(col('prop') > 1, 1).otherwise(col('prop')))

        # f. Filter exang and slope to be within specific sets
        dt = dt.filter(col('exang').isin([0, 1]))
        dt = dt.filter(col('slope').isin([1, 2, 3]))

        return dt


###### ML Pipeline ######
def pipeline(data: DataFrame):
    # Step 1: Data Cleaning
    cleaner = DataCleaner()
    cleaned_data = cleaner.transform(data)

    # Step 2: Feature Engineering
    feature_columns = [col for col in cleaned_data.columns if col != 'num']
    assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

    # Step 3: Define Classifiers
    lr = LogisticRegression(featuresCol='features', labelCol='num')
    rf = RandomForestClassifier(featuresCol='features', labelCol='num')

    # Step 4: Pipeline Setup
    pipeline_lr = Pipeline(stages=[assembler, lr])
    pipeline_rf = Pipeline(stages=[assembler, rf])

    # Step 5: Hyperparameter Tuning Setup
    paramGrid_lr = ParamGridBuilder() \
        .addGrid(lr.regParam, [0.01, 0.1, 0.5]) \
        .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
        .build()

    paramGrid_rf = ParamGridBuilder() \
        .addGrid(rf.numTrees, [10, 20, 50]) \
        .addGrid(rf.maxDepth, [5, 10, 20]) \
        .build()

    # Step 6: Cross-Validation Setup
    evaluator = BinaryClassificationEvaluator(labelCol='num', rawPredictionCol="rawPrediction", metricName='areaUnderROC')
    cv_lr = CrossValidator(estimator=pipeline_lr, estimatorParamMaps=paramGrid_lr, evaluator=evaluator, numFolds=NUMBER_OF_FOLDS, seed=SPLIT_SEED)
    cv_rf = CrossValidator(estimator=pipeline_rf, estimatorParamMaps=paramGrid_rf, evaluator=evaluator, numFolds=NUMBER_OF_FOLDS, seed=SPLIT_SEED)

    # Step 7: Train-Test Split
    train_data, test_data = cleaned_data.randomSplit([TRAIN_TEST_SPLIT, 1-TRAIN_TEST_SPLIT], seed=SPLIT_SEED)

    # Step 8: Fit the models on the training data
    model_lr = cv_lr.fit(train_data)
    model_rf = cv_rf.fit(train_data)

    # Step 9: Model Evaluation on test data
    predictions_lr = model_lr.transform(test_data)
    predictions_rf = model_rf.transform(test_data)

    auc_lr = evaluator.evaluate(predictions_lr)
    auc_rf = evaluator.evaluate(predictions_rf)

    # Step 10: Select the better model based on AUC and print the best parameters
    if auc_lr > auc_rf:
        best_model = model_lr
        best_auc = auc_lr
        best_model_name = "Logistic Regression"
        best_params = best_model.bestModel.stages[-1].extractParamMap()
    else:
        best_model = model_rf
        best_auc = auc_rf
        best_model_name = "Random Forest"
        best_params = best_model.bestModel.stages[-1].extractParamMap()

    # Print the details of the best model
    print(f"The best model is: {best_model_name}")
    print(f"AUC of the best model: {best_auc}")
    print("Best parameters:")
    for param, value in best_params.items():
        print(f"{param}: {value}")



###### Main Class ######
def main():
    # Create a Spark session
    spark = SparkSession.builder \
        .appName("Predict Heart Disease") \
        .getOrCreate()

    data = read_data(spark)
    pipeline(data)

    spark.stop()

main()
