# %%

import findspark
findspark.init()

from pyspark.sql import SparkSession

PARTITIONS = 8

spark = (
    SparkSession.builder
    .appName("Human Activity Recognition")
    .getOrCreate()
)

from load import load_partitioned_data
from pipeline import HARPipelineBuilder
from pyspark.sql.functions import col

train = load_partitioned_data(spark, '../data/partitioned/training', PARTITIONS).sample(False, 0.005, seed=123)

n_classes = train.filter(col("action").isNotNull()).select("action").distinct().count()

pipeline = HARPipelineBuilder(cap=20, 
                              features_size=100, 
                              hidden_layer = 64,
                              epochs=100,
                              batch_size=64,
                              n_classes=n_classes).build()

model = pipeline.fit(train)

model.save('../model/trained_pipeline')

# %%
