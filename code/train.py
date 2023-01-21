# %%

import findspark
from pyspark.sql import SparkSession
from load import load_partitioned_data
from pipeline import HARPipelineBuilder
from pyspark.sql.functions import col

findspark.init()

PARTITIONS = 8

spark = (
    SparkSession.builder
    .appName("Human Activity Recognition")
    .getOrCreate()
)

train = load_partitioned_data(spark, '/data/partitioned/training', PARTITIONS)\
    .sample(False, 0.005, seed=123) # Temporary sample for quick training

# Get the different amount of classes we have on training
n_classes = train.filter(col("action").isNotNull()).select("action").distinct().count()

# Create the full pipeline and train the model
pipeline = HARPipelineBuilder(cap=20, 
                              features_size=100, 
                              hidden_layer = 64,
                              epochs=100,
                              n_classes=n_classes).build()

model = pipeline.fit(train)

# Save the new version of the trained model
model.write().overwrite().save('../model/trained_pipeline')

# %%
