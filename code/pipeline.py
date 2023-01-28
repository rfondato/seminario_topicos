#%% 

from pyspark import keyword_only
from pyspark.ml import Transformer
from pyspark.ml.param.shared import HasInputCol, HasInputCols, HasOutputCol, HasOutputCols, Param, Params, TypeConverters
from pyspark.ml.util import DefaultParamsReadable, DefaultParamsWritable  
import pyspark.sql.functions as F
from pyspark.sql.types import FloatType, DoubleType
from pyspark.sql import Window
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, MinMaxScaler, StringIndexer, PCA
from itertools import product, chain


class FilterNulls(Transformer, DefaultParamsReadable, DefaultParamsWritable):
    """
        Filter all rows that have at least 1 null value (column).
    """

    @keyword_only
    def __init__(self):
        super(FilterNulls, self).__init__()
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, dataset):
        return dataset.dropna(how="any")


class CapValues(Transformer, HasInputCols, DefaultParamsReadable, DefaultParamsWritable):
    """
        Caps the input columns to the threshold value.
    """

    threshold = Param(Params._dummy(), "threshold", "cap absolute values greater than this threshold, to this threshold", typeConverter=TypeConverters.toInt)
    
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

        cap_value = F.udf(lambda x: threshold if x > threshold else -threshold if x < -threshold else x, FloatType())

        output = dataset
        for c in inputCols:
            output = output.withColumn(c, cap_value(c))

        return output

class EventSeparator (Transformer, HasInputCol, HasOutputCol, DefaultParamsReadable, DefaultParamsWritable):
    """
        Separate the rows in events, when the input value column changes for each window
        determined by partitionBy and orderBy columns.
        Store the event id on the output column.
    """

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

    def _transform(self, dataset):
        partitionBy = self.getPartitionBy()
        orderBy = self.getOrderBy()
        inputCol = self.getInputCol()
        outputCol = self.getOutputCol()

        mywindow = Window.partitionBy([F.col(x) for x in partitionBy])\
            .orderBy([F.col(x) for x in orderBy])

        return dataset\
            .withColumn("indicator", (F.col(inputCol) != F.lag(inputCol).over(mywindow)).cast("int"))\
            .fillna(0, subset=[ "indicator"])\
            .withColumn(outputCol, F.sum(F.col("indicator")).over(mywindow.rangeBetween(Window.unboundedPreceding, 0)))\
            .drop(F.col("indicator"))

class FeaturesGenerator (Transformer, HasInputCols, DefaultParamsReadable, DefaultParamsWritable):
    """
        Calculate a moving avg of period movingAvgPeriod over each input column,
        then generate windowSize lagged values for each.
        partitionBy and orderBy columns will be used as a Window on which to apply such transformations.
    """

    windowSize = Param(Params._dummy(), "windowSize", "amount of previous values to include on features vector", typeConverter=TypeConverters.toInt)
    movingAvgPeriod = Param(Params._dummy(), "movingAvgPeriod", "calculate moving avg with this period over input columns", typeConverter=TypeConverters.toInt)
    partitionBy = Param(Params._dummy(), "partitionBy", "column names to partition by, to create the lagged values", typeConverter=TypeConverters.toListString)
    orderBy = Param(Params._dummy(), "orderBy", "column names to order by, to create the lagged values", typeConverter=TypeConverters.toListString)

    @keyword_only
    def __init__(self, inputCols=[], windowSize=50, movingAvgPeriod=20, partitionBy=[], orderBy=[]):
        super(FeaturesGenerator, self).__init__()
        self._setDefault(inputCols=[], windowSize=50, movingAvgPeriod=20, partitionBy=[], orderBy=[])
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCols=[], windowSize=50, movingAvgPeriod=20, partitionBy=[], orderBy=[]):
        kwargs = self._input_kwargs
        return self._set(**kwargs)
   
    def setWindowSize(self, value):
        return self._set(windowSize=value)

    def getWindowSize(self):
        return self.getOrDefault(self.windowSize)

    def setMovingAvgPeriod(self, value):
        return self._set(movingAvgPeriod=value)

    def getMovingAvgPeriod(self):
        return self.getOrDefault(self.movingAvgPeriod)

    def setPartitionBy(self, partitionBy):
        return self._set(partitionBy=partitionBy)

    def getPartitionBy(self):
        return self.getOrDefault(self.partitionBy)
    
    def setOrderBy(self, orderBy):
        return self._set(orderBy=orderBy)

    def getOrderBy(self):
        return self.getOrDefault(self.orderBy)

    def _transform(self, dataset):
        windowSize = self.getWindowSize()
        partitionBy = self.getPartitionBy()
        orderBy = self.getOrderBy()
        inputCols = self.getInputCols()
        movingAvgPeriod = self.getMovingAvgPeriod()

        w = Window.partitionBy([F.col(x) for x in partitionBy])\
            .orderBy([F.col(x) for x in orderBy])

        avgWindow = w.rowsBetween(-movingAvgPeriod, 0)

        for c in inputCols:
            dataset = dataset.withColumn(c, F.avg(F.col(c)).over(avgWindow))

        lag_cols = [F.lag(F.col(c), i).over(w).alias(f"{c}_{i}") for (c,i) in product(inputCols, range(1, windowSize+1))]

        return dataset.select("*", *lag_cols)


class RowsSelector (Transformer, DefaultParamsReadable, DefaultParamsWritable):

    """
        Select one row every step rows inside each Window determined by partitionBy and orderBy.
    """

    step = Param(Params._dummy(), "step", "size of step between contiguous rows to take", typeConverter=TypeConverters.toInt)
    partitionBy = Param(Params._dummy(), "partitionBy", "column names to partition by, to create the lagged values", typeConverter=TypeConverters.toListString)
    orderBy = Param(Params._dummy(), "orderBy", "column names to order by, to create the lagged values", typeConverter=TypeConverters.toListString)

    @keyword_only
    def __init__(self, step=20, partitionBy=[], orderBy=[]):
        super(RowsSelector, self).__init__()
        self._setDefault(step=20, partitionBy=[], orderBy=[])
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, step=20, partitionBy=[], orderBy=[]):
        kwargs = self._input_kwargs
        return self._set(**kwargs)
   
    def setStep(self, value):
        return self._set(step=value)

    def getStep(self):
        return self.getOrDefault(self.step)

    def setPartitionBy(self, partitionBy):
        return self._set(partitionBy=partitionBy)

    def getPartitionBy(self):
        return self.getOrDefault(self.partitionBy)
    
    def setOrderBy(self, orderBy):
        return self._set(orderBy=orderBy)

    def getOrderBy(self):
        return self.getOrDefault(self.orderBy)

    def _transform(self, dataset):
        step = self.getStep()
        partitionBy = self.getPartitionBy()
        orderBy = self.getOrderBy()

        w = Window.partitionBy([F.col(x) for x in partitionBy])\
            .orderBy([F.col(x) for x in orderBy])

        return dataset.withColumn("rank", F.rank().over(w)).filter((F.col("rank") % F.lit(step)) == 0)

class ColumnsSelector (Transformer, HasOutputCols, DefaultParamsReadable, DefaultParamsWritable):

    """
        Select only the outputCols from the input dataset.
    """

    @keyword_only
    def __init__(self, outputCols=[]):
        super(ColumnsSelector, self).__init__()
        self._setDefault(outputCols=[])
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, outputCols=[]):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, dataset):
        return dataset.select(self.getOutputCols())

class ClassWeightBalancer (Transformer, HasInputCol, HasOutputCol, DefaultParamsReadable, DefaultParamsWritable):

    """
        Create an output column with the weight of each class
    """

    @keyword_only
    def __init__(self, inputCol="label", outputCol="weight"):
        super(ClassWeightBalancer, self).__init__()
        self._setDefault(inputCol="label", outputCol="weight")
        kwargs = self._input_kwargs
        self.setParams(**kwargs)

    @keyword_only
    def setParams(self, inputCol="label", outputCol="weight"):
        kwargs = self._input_kwargs
        return self._set(**kwargs)

    def _transform(self, dataset):
        inputCol = self.getInputCol()
        outputCol = self.getOutputCol()

        count_by_label = dataset.select(inputCol).groupBy(inputCol).count().collect()
        labels = [x[inputCol] for x in count_by_label]

        # Short circuit if predicting no data
        if len(labels) <= 0:
            return dataset

        counts = [x["count"] for x in count_by_label]
        weights = [sum(counts) / c for c in counts] # Inverse of proportion
        class_weights = {l: w for l, w in zip(labels, weights)}
        
        # Normalize weights, so that most frequent label has weight 1
        min_weight = min(class_weights.values())
        class_weights = {k: (class_weights[k] / min_weight) for k in class_weights.keys() }

        # These 2 lines are taken from: https://danvatterott.com/blog/2019/11/18/balancing-model-weights-in-pyspark/
        mapping_expr = F.create_map([F.lit(x) for x in chain(*class_weights.items())])
        return dataset.withColumn(outputCol, mapping_expr.getItem(F.col(inputCol)))


from pyspark.ml.classification import RandomForestClassifier

def create_estimator(numTrees=20, maxDepth=5, featuresCol="features", weightCol="weight"):
    return RandomForestClassifier(numTrees=numTrees, maxDepth=maxDepth, featuresCol=featuresCol, weightCol=weightCol, seed=123)

class HARPipelineBuilder:

    """
        Represents a pipeline model for HAR (Human Activity Recognition)
    """

    def __init__(self, 
                cap = 20, 
                features_size = 50,
                moving_avg_period = 20,
                slide_between_windows = 20,
                pca_components = 10,
                num_trees=20,
                max_depth=5) -> None:
        """
            cap: Threshold to cap the input measures to. Default 20.
            features_size: Amount of previous values to use as features, for each row.
            moving_avg_period: Period of the moving avg to apply to the input measures.
            slide_between_windows: Select only one row every this value.
            pca_components: Select the first pca_components dimensions, after applying PCA.
            num_trees: Number of trees to use by the random forest estimator.
            max_depth: Max depth of each tree used by the random forest estimator.
        """
        self.cap = cap
        self.features_size = features_size
        self.moving_avg_period = moving_avg_period
        self.step = slide_between_windows
        self.pca_k = pca_components
        self.num_trees = num_trees
        self.max_depth = max_depth

    def build(self):
        """
            Builds and returns a HAR pipeline to be fit and then used to transform new data.
        """
        str_indexer = StringIndexer(inputCol="action", outputCol="label", handleInvalid="keep")
        filter_nulls = FilterNulls()
        cap_values = CapValues(inputCols=["x", "y", "z"], threshold=self.cap)
        events_separator = EventSeparator(inputCol="action", outputCol="event", partitionBy=["userId"], orderBy=["timestamp"])
        features_generator = FeaturesGenerator(inputCols=["x", "y", "z"], windowSize=self.features_size, movingAvgPeriod=self.moving_avg_period, partitionBy=["userId", "event"], orderBy=["timestamp"])
        
        feature_cols = [f'{c}_{i}' for (c,i) in product(["x", "y", "z"], range(1, self.features_size + 1))]
        
        vector_assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
        pca = PCA(k=self.pca_k, inputCol="features", outputCol="features_PCA")
        scaler = MinMaxScaler(inputCol="features_PCA", outputCol="features_scaled")
        rows_selector = RowsSelector(step = self.step, partitionBy=["userId", "event"], orderBy=["timestamp"])
        class_balancer = ClassWeightBalancer(inputCol="label", outputCol="weight")
        estimator = create_estimator(numTrees=self.num_trees, maxDepth=self.max_depth, featuresCol="features_scaled", weightCol="weight")

        return Pipeline(stages=[filter_nulls, # Get rid of rows with at least 1 null column
                                str_indexer, # Convert actions to numeric labels for training
                                cap_values, # Cap values to the [-cap,+cap] interval
                                events_separator, # Separate contiguous rows of the same user and action into events
                                features_generator, # Generate main features for training (lagged values of avg of x,y,z , up to features_size)
                                filter_nulls, # Get rid of rows that have null values on lagged columns (first feature_size cols per event)
                                vector_assembler, # Make a "features" cols for ML training combining all lagged cols into a vector
                                ColumnsSelector(outputCols=["userId", "timestamp", "event", "label", "features"]), # Select only needed cols
                                rows_selector, # Select only one row per contiguous "step" rows.
                                pca, # Apply PCA to reduce dimensionality of the feature vectors
                                scaler, # Min Max scaling for the "features" vectors
                                ColumnsSelector(outputCols=["userId", "timestamp", "label", "features_scaled"]), # Get rid of extra feature cols
                                class_balancer, # Calculate weights of each class according to the frecuency over the dataset
                                estimator # Random Forest estimator to predict the actions performed by the users
                                ])

# %%
