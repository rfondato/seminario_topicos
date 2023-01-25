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

    #.withColumn("value", F.encode(F.col("value"), "iso-8859-1").cast("binary"))\
    #.withColumn("key", F.encode(F.col("key"), "iso-8859-1").cast("binary"))\

# Read from real_time parquet table (which will be populated by sample) as a stream,
# and write to the HAR kafka topic
load_stream(spark, '/data/real_time')\
    .withColumn("value", F.to_json(F.struct(F.col("*")) ) )\
    .withColumn("key", F.col("timestamp").cast("string"))\
    .writeStream\
    .format("kafka")\
    .option("kafka.bootstrap.servers", KAFKA_BOOTSTRAP_SERVERS)\
    .option("topic", KAFKA_TOPIC)\
    .option("checkpointLocation", "/data/checkpoint/producer")\
    .start()\
    .awaitTermination()

# %%
