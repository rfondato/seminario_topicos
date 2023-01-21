# %%

import findspark
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.mllib.evaluation import MulticlassMetrics
from load import load_partitioned_data
import mlflow

findspark.init()

PARTITIONS = 8

spark = (
    SparkSession.builder
    .appName("Human Activity Recognition")
    .getOrCreate()
)

# Load the test partition
test = load_partitioned_data(spark, '../data/partitioned/testing', PARTITIONS)

# Load the trained model
model = PipelineModel.load('../model/trained_pipeline')

predictions = model.transform(test)

predictions_and_labels = predictions.select("prediction", "label").rdd.map(lambda r: (float(r[0]), float(r[1])))
metrics = MulticlassMetrics(predictions_and_labels)

mlflow.set_tracking_uri("http://localhost:5000")
experiment = mlflow.set_experiment("TP Seminario - HAR")

with mlflow.start_run():
    mlflow.log_metric("accuracy", metrics.accuracy)
    mlflow.log_metric ("precision", metrics.precision())
    mlflow.log_metric("recall", metrics.recall())
    mlflow.spark.log_model(model, "spark-model")

# %%
