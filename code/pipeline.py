#%% 

from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasInputCols, HasOutputCol, Param, Params, TypeConverters
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable  
from pyspark.sql.functions import col, abs, udf
from pyspark.sql.types import DoubleType
from pyspark.ml import Pipeline

class FilterNulls(Transformer, DefaultParamsReadable, DefaultParamsWritable):

    def __init__(self):
        super(FilterNulls, self).__init__()

    def _transform(self, dataset):
        return dataset.dropna(how="any")


class CapValues(Transformer, HasInputCols, DefaultParamsReadable, DefaultParamsWritable):

    threshold = Param(Params._dummy(), "threshold", "cap abs values greater than this threshold, to this threshold", typeConverter=TypeConverters.toInt)
    
    @keyword_only
    def __init__(self, inputCols=[], threshold=20):
        super(CapValues, self).__init__()
        self._setDefault(inputCols=[], threshold=20)
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCols=[], threshold=20):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setThreshold(self, value):
        return self._set(threshold=value)

    def getThreshold(self):
        return self.getOrDefault(self.threshold)

    def setInputCols(self, value):
        return self._set(inputCols=value)

    def _transform(self, dataset):
        threshold = self.getThreshold()
        inputCols = self.getInputCols()

        cap_value = udf(lambda x: threshold if x > threshold else -threshold if x < -threshold else x, DoubleType())

        output = dataset
        for c in inputCols:
            output = output.withColumn(c, cap_value(c))

        return output

from pyspark.sql.functions import lag, col, sum
from pyspark.sql import Window
from pyspark.ml.feature import VectorAssembler

class EventSeparator (Transformer, HasInputCol, HasOutputCol, DefaultParamsReadable, DefaultParamsWritable):

    partitionBy = Param(Params._dummy(), "partitionBy", "column names to partition by, to create the events", typeConverter=TypeConverters.toListString)
    orderBy = Param(Params._dummy(), "orderBy", "column names to order by, to create the events", typeConverter=TypeConverters.toListString)

    @keyword_only
    def __init__(self, inputCol="", outputCol="features", partitionBy=[], orderBy=[]):
        super(EventSeparator, self).__init__()
        self._setDefault(inputCol="", outputCol="features", partitionBy=[], orderBy=[])
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol="", outputCol="features", partitionBy=[], orderBy=[]):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def setPartitionBy(self, partitionBy):
        return self._set(partitionBy=partitionBy)

    def getPartitionBy(self):
        return self.getOrDefault(self.partitionBy)
    
    def setOrderBy(self, orderBy):
        return self._set(orderBy=orderBy)

    def getOrderBy(self):
        return self.getOrDefault(self.orderBy)

    def transform(self, dataset):
        partitionBy = self.getPartitionBy()
        orderBy = self.getOrderBy()
        inputCol = self.getInputCol()
        outputCol = self.getOutputCol()

        mywindow = Window.partitionBy([col(x) for x in partitionBy])\
            .orderBy([col(x) for x in orderBy])

        return dataset\
            .withColumn("indicator", (col(inputCol) != lag(inputCol).over(mywindow)).cast("int"))\
            .fillna(0, subset=[ "indicator"])\
            .withColumn(outputCol, sum(col("indicator")).over(mywindow.rangeBetween(Window.unboundedPreceding, 0)))\
            .drop(col("indicator"))

class FeaturesVectorGenerator (Transformer, HasInputCols, HasOutputCol, DefaultParamsReadable, DefaultParamsWritable):

    windowSize = Param(Params._dummy(), "windowSize", "amount of previous values to include on features vector", typeConverter=TypeConverters.toInt)
    partitionBy = Param(Params._dummy(), "partitionBy", "column names to partition by, to create the lagged values", typeConverter=TypeConverters.toListString)
    orderBy = Param(Params._dummy(), "orderBy", "column names to order by, to create the lagged values", typeConverter=TypeConverters.toListString)

    @keyword_only
    def __init__(self, inputCols=[], outputCol="features", windowSize=50, partitionBy=[], orderBy=[]):
        super(FeaturesVectorGenerator, self).__init__()
        self._setDefault(inputCols=[], outputCol="features", windowSize=50, partitionBy=[], orderBy=[])
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCols=[], outputCol="features", windowSize=50, partitionBy=[], orderBy=[]):
        kwargs = self._input_kwargs
        return self._set(**kwargs)
   
    def setWindowSize(self, value):
        return self._set(windowSize=value)

    def getWindowSize(self):
        return self.getOrDefault(self.windowSize)

    def setPartitionBy(self, partitionBy):
        return self._set(partitionBy=partitionBy)

    def getPartitionBy(self):
        return self.getOrDefault(self.partitionBy)
    
    def setOrderBy(self, orderBy):
        return self._set(orderBy=orderBy)

    def getOrderBy(self):
        return self.getOrDefault(self.orderBy)

    def transform(self, dataset):
        windowSize = self.getWindowSize()
        partitionBy = self.getPartitionBy()
        orderBy = self.getOrderBy()
        inputCols = self.getInputCols()
        outputCol = self.getOutputCol()

        mywindow = Window.partitionBy([col(x) for x in partitionBy])\
            .orderBy([col(x) for x in orderBy])

        output = dataset

        genCols = []

        for f in inputCols:
            for i in range(windowSize):
                lagCol = f + '_' + str(i+1)
                output = output.withColumn(lagCol, lag(output[f], i+1).over(mywindow))
                genCols.append(lagCol)

        assembler = VectorAssembler(inputCols=genCols,outputCol=outputCol)

        return Pipeline(stages=[FilterNulls(), assembler]).fit(output).transform(output).select(["userId", "action", "features", "label"])

from pyspark.ml.classification import MultilayerPerceptronClassifier, RandomForestClassifier

def create_estimator(features_size, hidden_layer=64, n_classes = 6, epochs=100, featuresCol="features"):
    return MultilayerPerceptronClassifier(layers=[features_size, hidden_layer, n_classes], seed=123, maxIter=epochs, featuresCol=featuresCol)

from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import StringIndexer

class HARPipelineBuilder:

    def __init__(self, 
                cap = 20, 
                features_size = 50,
                hidden_layer = 64,
                n_classes = 6,
                epochs=100) -> None:
        self.cap = cap
        self.features_size = features_size
        self.hidden_layer = hidden_layer
        self.n_classes = n_classes
        self.epochs = epochs

    def build(self):
        str_indexer = StringIndexer(inputCol="action", outputCol="label")
        filter_nulls = FilterNulls()
        cap_values = CapValues(inputCols=["x", "y", "z"], threshold=self.cap)
        events_separator = EventSeparator(inputCol="action", outputCol="event", partitionBy=["userId"], orderBy=["timestamp"])
        features_vector_gen = FeaturesVectorGenerator(inputCols=["x", "y", "z"], outputCol="features", windowSize=self.features_size, partitionBy=["userId", "event"], orderBy=["timestamp"])
        scaler = MinMaxScaler(inputCol="features", outputCol="features_scaled")
        estimator = create_estimator(self.features_size * 3, hidden_layer=self.hidden_layer, n_classes=self.n_classes, featuresCol="features_scaled")
        return Pipeline(stages=[filter_nulls, # Get rid of rows with nulls
                                str_indexer, # Convert actions to labels
                                cap_values, # Cap values to the [-cap,+cap] interval
                                events_separator, # Separate contiguous rows of the same action and user into events
                                features_vector_gen, # Generate vector of features (lagged values of x,y,z , up to features_size)
                                filter_nulls, # Now filter rows that have nulls for some lagged values
                                scaler, # Min Max scaling for x,y,z values
                                estimator # Neural Network to estimate the action performed by the user
                                ])
