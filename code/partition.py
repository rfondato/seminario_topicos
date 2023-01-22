# %%

import findspark
findspark.init()

from pyspark.sql import SparkSession
from load import load_accelerometer_data
import pyspark.sql.functions as F

spark = (
    SparkSession.builder
    .appName("TP Seminario HAR")
    .getOrCreate()
)

threshold = 1607 # This threshold is the userId we will use to split training and testing. Was calculated on EDA notebook.

# Load already labeled raw data from csv file
ad = load_accelerometer_data(spark, '/data/WISDM_at_v2.0_raw.txt')

# Partition training data and save to parquet
ad.filter(F.col("userId") <= F.lit(threshold))\
    .repartition("userId", "action")\
    .write\
    .option("header", True)\
    .partitionBy("userId", "action")\
    .mode("overwrite")\
    .parquet('/data/partitioned/training')

# Partition testing data and save to parquet
ad.filter(F.col("userId") > F.lit(threshold))\
    .repartition("userId", "action")\
    .write\
    .option("header", True)\
    .partitionBy("userId", "action")\
    .mode("overwrite")\
    .parquet('/data/partitioned/testing')
