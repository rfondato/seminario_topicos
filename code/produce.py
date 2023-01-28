# %%

import findspark
from pyspark.sql import SparkSession
from load import load_stream
import pyspark.sql.functions as F

KAFKA_BOOTSTRAP_SERVERS = "kafka:9092"
KAFKA_TOPIC = "HAR"

findspark.init()

spark = (
    SparkSession.builder
    .appName("TP Seminario - Kafka Producer")
    .getOrCreate()
)

# Read from real_time parquet table (which will be populated by sample) as a stream,
# and write to the HAR kafka topic, every 1 minute.
load_stream(spark, '/data/real_time')\
    .withColumn("value", F.to_json(F.struct(F.col("*")) ) )\
    .withColumn("key", F.col("timestamp").cast("string"))\
    .writeStream\
    .format("kafka")\
    .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS)\
    .option("topic", KAFKA_TOPIC)\
    .option("checkpointLocation", "/data/checkpoint/producer")\
    .trigger(processingTime="1 minute")\
    .start()\
    .awaitTermination()

# %%
