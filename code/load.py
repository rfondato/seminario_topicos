from pyspark.sql.types import LongType, IntegerType, StringType, StructType, DoubleType
from pyspark.sql.functions import col, regexp_replace

def get_accelerometer_schema():
    return StructType()\
        .add("userId", IntegerType(), True)\
        .add("action", StringType(), True)\
        .add("timestamp", LongType(), True)\
        .add("x", DoubleType(), True)\
        .add("y", DoubleType(), True)\
        .add("z", StringType(), True)

def get_demographic_schema():
    return StructType()\
        .add("userId", IntegerType(), True)\
        .add("height", DoubleType(), True)\
        .add("sex", StringType(), True)\
        .add("age", IntegerType(), True)\
        .add("weight", DoubleType(), True)\
        .add("leg_injury", IntegerType(), True)

def load_accelerometer_data(sparkSession, file):
    return sparkSession.read.csv(file, schema=get_accelerometer_schema())\
        .withColumn("z", regexp_replace(col("z"), ';', '').cast(DoubleType()))

def load_demographic_data(sparkSession, file):
    return sparkSession.read.csv(file, schema=get_demographic_schema())

def load_partitioned_data(sparkSession, base, partitions):
    return sparkSession.read.parquet(base).coalesce(partitions)
