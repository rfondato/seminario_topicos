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
    .appName("TP Seminario - Sample Data")
    .getOrCreate()
)

# Load the unlabeled data
unlabeled_data = load_accelerometer_data(spark, '/data/WISDM_at_v2.0_unlabeled_raw.txt')

SAMPLES = 5

# Pick a number of samples from it.
samples = unlabeled_data.rdd.takeSample(False, SAMPLES, seed=None) 

for s in samples:
    # Get the userId and timestamp
    userId = s[0]
    timestamp = s[2]

    # Data was sampled at 20hz, so 1 sample every 50 ms and timestamp is in ms, then this means 1 minute = 1200 rows
    TIMESTAMP_RANGE = 60000

    # Using the random user Id and timestamp, take a range of 1 particular action to sample (-30 seconds to +30 seconds)
    to_predict = unlabeled_data\
        .filter((F.col("userid") == F.lit(userId)) & (F.lit(timestamp - TIMESTAMP_RANGE / 2) <= F.col("timestamp")) & (F.col("timestamp") <= F.lit(timestamp + TIMESTAMP_RANGE / 2)))

    # Append samples to parquet real_time data
    to_predict\
        .write\
        .option("header", True)\
        .mode("append")\
        .parquet('/data/real_time')

# %%
