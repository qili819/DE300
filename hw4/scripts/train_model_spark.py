from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
import json

class TrainModelSpark:
    def __init__(self, spark: SparkSession):
        self.spark = spark

    def train_spark(self, input_path: str):
        # Read the cleaned data
        data = self.spark.read.csv(input_path, header=True, inferSchema=True)

        # Feature Engineering
        feature_columns = [col for col in data.columns if col != 'num']
        assembler = VectorAssembler(inputCols=feature_columns, outputCol="features")

        # Define Classifiers
        lr = LogisticRegression(featuresCol='features', labelCol='num')
        rf = RandomForestClassifier(featuresCol='features', labelCol='num')

        # Pipeline Setup
        pipeline_lr = Pipeline(stages=[assembler, lr])
        pipeline_rf = Pipeline(stages=[assembler, rf])

        # Hyperparameter Tuning Setup
        paramGrid_lr = ParamGridBuilder() \
            .addGrid(lr.regParam, [0.01, 0.1, 0.5]) \
            .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
            .build()

        paramGrid_rf = ParamGridBuilder() \
            .addGrid(rf.numTrees, [10, 20, 50]) \
            .addGrid(rf.maxDepth, [5, 10, 20]) \
            .build()

        # Cross-Validation Setup
        evaluator = BinaryClassificationEvaluator(labelCol='num', rawPredictionCol="rawPrediction", metricName='areaUnderROC')
        cv_lr = CrossValidator(estimator=pipeline_lr, estimatorParamMaps=paramGrid_lr, evaluator=evaluator, numFolds=5, seed=7576)
        cv_rf = CrossValidator(estimator=pipeline_rf, estimatorParamMaps=paramGrid_rf, evaluator=evaluator, numFolds=5, seed=7576)

        # Train-Test Split
        train_data, test_data = data.randomSplit([0.9, 0.1], seed=7576)

        # Fit the models on the training data
        model_lr = cv_lr.fit(train_data)
        model_rf = cv_rf.fit(train_data)

        # Model Evaluation on test data
        predictions_lr = model_lr.transform(test_data)
        predictions_rf = model_rf.transform(test_data)

        auc_lr = evaluator.evaluate(predictions_lr)
        auc_rf = evaluator.evaluate(predictions_rf)

        best_log_reg = {"best_params": model_lr.bestModel.stages[-1].extractParamMap(), "best_score": auc_lr}
        best_rf = {"best_params": model_rf.bestModel.stages[-1].extractParamMap(), "best_score": auc_rf}

        with open("/tmp/best_log_reg_spark.json", "w") as f:
            json.dump(best_log_reg, f)

        with open("/tmp/best_rf_spark.json", "w") as f:
            json.dump(best_rf, f)
