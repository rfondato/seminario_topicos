# %%

import findspark
findspark.init()

from pyspark.sql import SparkSession

CORES = 4

spark = (
    SparkSession.builder
    .appName("Human Activity Recognition")
    .getOrCreate()
)

from load import load_accelerometer_data

import pyspark.sql.functions as F

threshold = 1607

ad = load_accelerometer_data(spark, '/data/WISDM_at_v2.0_raw.txt')

# Partition training data
ad.filter(F.col("userId") <= F.lit(threshold))\
    .repartition("userId", "action")\
    .write\
    .option("header", True)\
    .partitionBy("userId", "action")\
    .mode("overwrite")\
    .parquet('/data/partitioned/training')

# Partition testing data
ad.filter(F.col("userId") > F.lit(threshold))\
    .repartition("userId", "action")\
    .write\
    .option("header", True)\
    .partitionBy("userId", "action")\
    .mode("overwrite")\
    .parquet('/data/partitioned/testing')
