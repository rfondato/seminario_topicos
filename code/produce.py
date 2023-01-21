# %%

import findspark
from pyspark.sql import SparkSession
from load import load_accelerometer_data
import pyspark.sql.functions as F

findspark.init()

spark = (
    SparkSession.builder
    .appName("Human Activity Recognition")
    .getOrCreate()
)

unlabeled_data = load_accelerometer_data(spark, '../data/WISDM_at_v2.0_unlabeled_raw.txt')

random_row = unlabeled_data.sample(False, 0.001).limit(1).collect()
userId = random_row[0][0]
timestamp = random_row[0][2]

to_predict = unlabeled_data.filter((F.col("userid") == F.lit(userId)) & (F.lit(timestamp) <= F.col("timestamp")) & (F.col("timestamp") <= F.lit(timestamp + 10000)))

# %%
