from pyspark.sql.types import LongType, IntegerType, StringType, StructType, FloatType
from pyspark.sql.functions import col, regexp_replace

# Using FloatType instead of double saves a lot of memory, considering that we will
# create few hundred lagged cols for x, y, z, at a cost of a lower precision.

def get_accelerometer_schema():
    return StructType()\
        .add("userId", IntegerType(), True)\
        .add("action", StringType(), True)\
        .add("timestamp", LongType(), True)\
        .add("x", FloatType(), True)\
        .add("y", FloatType(), True)\
        .add("z", StringType(), True)

def get_realtime_schema():
    return StructType()\
        .add("userId", IntegerType(), True)\
        .add("action", StringType(), True)\
        .add("timestamp", LongType(), True)\
        .add("x", FloatType(), True)\
        .add("y", FloatType(), True)\
        .add("z", FloatType(), True)

def get_demographic_schema():
    return StructType()\
        .add("userId", IntegerType(), True)\
        .add("height", FloatType(), True)\
        .add("sex", StringType(), True)\
        .add("age", IntegerType(), True)\
        .add("weight", FloatType(), True)\
        .add("leg_injury", IntegerType(), True)

def load_accelerometer_data(sparkSession, file):
    return sparkSession.read.csv(file, schema=get_accelerometer_schema())\
        .withColumn("z", regexp_replace(col("z"), ';', '').cast(FloatType()))

def load_demographic_data(sparkSession, file):
    return sparkSession.read.csv(file, schema=get_demographic_schema())

def load_partitioned_data(sparkSession, base, partitions):
    return sparkSession.read.parquet(base).coalesce(partitions)

def load_stream(sparkSession, file):
    return sparkSession.readStream.schema(get_realtime_schema()).parquet(file)
