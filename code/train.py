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

PARTITIONS = 8

spark = (
    SparkSession.builder
    .appName("TP Seminario HAR")
    .getOrCreate()
)

# Load partitioned training data. We will reduce partitions up to the number of cores we have available.
train = load_partitioned_data(spark, '/data/partitioned/training', PARTITIONS)

# Get the different amount of classes we have on training
n_classes = train.filter(col("action").isNotNull()).select("action").distinct().count()

# Create the full pipeline and train the model
pipeline = HARPipelineBuilder(cap=20, # Per docs, sensor values should be between [-20, 20]
                             features_size=100, # Window of 5 seconds to determine activity
                             hidden_layer = 64,
                             moving_avg_period=20, # 1 second of avg window
                             slide_between_windows=20, # slide 1 second at a time per event
                             epochs=1,
                             n_classes=n_classes).build()

# Train the model using the training dataset
model = pipeline.fit(train)

# Save the new version of the trained model
model.write().overwrite().save('/model/trained_pipeline')

# After training, add the flag file that indicates that the model completed training
Path('/model/.created').touch()

# %%
