# %%

import findspark
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel
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

output_path = "/data/predictions/"
checkpoint_path = "/data/checkpoint/consumer"

# Select only the extracted content
realtime_df = json_content.select("content.*")

# Use the model to transform the realtime data and get the predictions
# Then save to csv every 30 seconds
#model.transform(realtime_df)\ 
# Non time based partitions non supported!! https://stackoverflow.com/questions/53294809/spark-non-time-based-windows-are-not-supported-on-streaming-dataframes-dataset

def process_batch(batch_df, batch_id):
    model.transform(batch_df)\
    .select("userId", "timestamp", "action", "label", "prediction")\
    .write\
    .option("header", True)\
    .mode("overwrite")\
    .csv(f"{output_path}pred_batch_{batch_id}")

realtime_df\
    .writeStream \
    .foreachBatch(process_batch)\
    .start()\
    .awaitTermination()\
    .stop()
    #.format("csv")\
    #.trigger(processingTime="30 seconds")\
    #.option("checkpointLocation", checkpoint_path)\
    #.option("path", output_path)\
    #.outputMode("append")\
    #.start()\
    #.awaitTermination()\
    #.stop()

# %%
