# %%

import findspark
from pyspark.sql import SparkSession
from load import load_partitioned_data
from pipeline import HARPipelineBuilder
from pyspark.sql.functions import col
import os
from pathlib import Path

# Before training, remove the flag file that indicates that the model completed training
if os.path.exists("/model/.created"):
  os.remove("/model/.created")

findspark.init()

CORES = 6
PARTITIONS = CORES * 8

spark = (
    SparkSession.builder
    .appName("TP Seminario HAR")
    .getOrCreate()
)

# Load partitioned training data. We will reduce partitions up to PARTITIONS value.
train = load_partitioned_data(spark, '/data/partitioned/training', PARTITIONS)

# Get the different amount of classes we have on training
n_classes = train.filter(col("action").isNotNull()).select("action").distinct().count()

# Create the full pipeline and train the model
pipeline = HARPipelineBuilder(cap=20, # Per docs, sensor values should be between [-20, 20]
                             features_size=100, # Window of 5 seconds to determine activity
                             hidden_layer = 64, # 64 neurons on the hidden layer of the estimator
                             moving_avg_period = 5, # 200 millisec of avg window to smooth measures
                             pca_components=10, # Only use the first 10 components that explain more variance as features
                             slide_between_windows=20, # slide 1 second at a time per event
                             epochs=100, # Train for this amount of epochs
                             n_classes=n_classes).build() # Set the number of classes to predict and build the pipeline

# Train the model using the training dataset
model = pipeline.fit(train)

# Save the new version of the trained model
model.write().overwrite().save('/model/trained_pipeline')

# After training, add the flag file that indicates that the model completed training
Path('/model/.created').touch()

# %%
