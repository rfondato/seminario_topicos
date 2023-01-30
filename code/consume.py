# %%

import findspark
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
from pyspark.ml.feature import IndexToString
from load import get_realtime_schema
import pyspark.sql.functions as F

KAFKA_BOOTSTRAP_SERVERS = "kafka:9092"
KAFKA_TOPIC = "HAR"

findspark.init()

spark = (
    SparkSession.builder
    .appName("TP Seminario - Kafka Consumer")
    .getOrCreate()
)

# Load the trained model
model = PipelineModel.load('/model/trained_pipeline')

# Prepare the index to string object, which will map label numbers back to string actions
indexToString = IndexToString(inputCol="prediction", outputCol="predicted_action", labels=model.stages[1].labels)

# Read the stream from kafka HAR topic
jsonStream = spark.readStream.format("kafka") \
    .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS) \
    .option("subscribe", KAFKA_TOPIC) \
    .option("startingOffsets", "earliest") \
    .load()

# Get the schema of the realtime data
schema = get_realtime_schema()

# Extract the data from the value column (a json column)
json_content = jsonStream.select(
    F.from_json(F.col("value").cast("string"), schema).alias("content")
)

checkpoint_path = "/data/checkpoint/consumer"
output_path = "/data/predictions/"

# Select only the extracted content
realtime_df = json_content.select("content.*")

# Use the model to transform the realtime data and get the predictions
# Process micro-batches of 5 minutes

def process_batch(batch_df, batch_id):

    predicitions = model.transform(batch_df)
    predicitions = indexToString.transform(predicitions)

    predicitions\
    .select("userId", "timestamp", "predicted_action")\
    .write\
    .option("header", True)\
    .mode("append")\
    .csv(f"{output_path}pred_batch_{batch_id}")

realtime_df\
    .writeStream \
    .option("checkpointLocation", checkpoint_path)\
    .trigger(processingTime="5 minutes")\
    .foreachBatch(process_batch)\
    .start()\
    .awaitTermination()\
    .stop()

# %%
