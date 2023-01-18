# %%

import findspark
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from load import load_partitioned_data

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
model = Pipeline.load('../model/trained_pipeline')

predictions = model.transform(test)

predictions.show()
# %%
