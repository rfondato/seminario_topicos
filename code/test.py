# %%

import findspark
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.ml import PipelineModel
from pyspark.mllib.evaluation import MulticlassMetrics
from load import load_partitioned_data
import mlflow
from datetime import datetime

findspark.init()

CORES = 6
PARTITIONS = CORES * 4

spark = (
    SparkSession.builder
    .appName("TP Seminario HAR")
    .getOrCreate()
)

# Load the test partition.  We will reduce partitions up to PARTITIONS value.
test = load_partitioned_data(spark, '/data/partitioned/testing', PARTITIONS)

# Load the trained model
model = PipelineModel.load('/model/trained_pipeline')

# Run the model to get the predicted labels
predictions = model.transform(test)

# Calculate number of classes we have on testing set
n_classes = predictions.filter(F.col("label").isNotNull()).select("label").distinct().count()

# Get the metrics of the model, by using MulticlassMetrics
predictions_and_labels = predictions.select("prediction", "label").rdd.map(lambda r: (float(r[0]), float(r[1])))
metrics = MulticlassMetrics(predictions_and_labels)

# Create the experiment on mlFlow local server
mlflow.set_tracking_uri("http://mlflow_server:5000")
experiment = mlflow.set_experiment("TP Seminario - HAR")

now_str = datetime.now().strftime("%m/%d/%Y %H:%M:%S")

# Connect to mlflow and track accuracy, precision, recall and the model artifact
with mlflow.start_run(run_name=f"HAR {now_str}"):
    mlflow.log_metric("Accuracy", metrics.accuracy)
    for i in range(n_classes):
        mlflow.log_metric(f"Precision class {i}", metrics.precision(i))
        mlflow.log_metric(f"Recall class {i}", metrics.recall(i))
    
    mlflow.spark.log_model(model, f"har-model-{now_str}")

# %%
