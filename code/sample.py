# %%

import findspark
from pyspark.sql import SparkSession
from load import load_accelerometer_data
import pyspark.sql.functions as F

KAFKA_BOOTSTRAP_SERVERS = "kafka:9092"
KAFKA_TOPIC = "HAR"

findspark.init()

spark = (
    SparkSession.builder
    .appName("TP Seminario HAR")
    .getOrCreate()
)

unlabeled_data = load_accelerometer_data(spark, '/data/WISDM_at_v2.0_unlabeled_raw.txt')
random_row = unlabeled_data.rdd.takeSample(False, 1, seed=None) 
userId = random_row[0][0]
timestamp = random_row[0][2]

# Data was sampled at 20hz, so 1 sample every 50 ms and timestamp is in ms, so this means 200 rows, which is 10 seconds 
TIMESTAMP_RANGE = 10000

# Using the random user Id and timestamp, take a range of 1 particular action to sample
to_predict = unlabeled_data\
    .filter((F.col("userid") == F.lit(userId)) & (F.lit(timestamp) <= F.col("timestamp")) & (F.col("timestamp") <= F.lit(timestamp + 10000)))\

print("userId: ", userId, "timestamp: ", timestamp)

# Append samples to parquet real_time data
to_predict\
    .write\
    .option("header", True)\
    .mode("append")\
    .parquet('/data/real_time')

# %%
