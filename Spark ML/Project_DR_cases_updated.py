# Databricks notebook source
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.sql import SparkSession
from pyspark.context import SparkContext

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler, StringIndexer, VectorIndexer, MinMaxScaler
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder, TrainValidationSplit
from pyspark.ml.evaluation import BinaryClassificationEvaluator

from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.regression import GBTRegressor
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.feature import OneHotEncoder
from pyspark.ml.evaluation import BinaryClassificationEvaluator, RegressionEvaluator
from pyspark.ml.classification import LogisticRegression


# COMMAND ----------

# DataFrame Schema, that should be a Table schema
coviddeath_schema = StructType([
  StructField('UID', IntegerType()),
  StructField('iso2', StringType()),
  StructField('iso3', StringType()),
  StructField('code3', IntegerType()),
  StructField('FIPS', IntegerType()),
  StructField('Admin2', StringType()),
  StructField('Lat', DoubleType()),
  StructField('CombinedKey', StringType()),
  StructField('Date1', StringType()),
  StructField('Case', IntegerType()),
  StructField('Long', DoubleType()),
  StructField('Country', StringType()),
  StructField('Province', StringType()),
  StructField('Temp', StringType()),
  StructField('Date', StringType()),
  StructField('Day', StringType()),
  StructField('Year', StringType())
  ])

# COMMAND ----------

IS_SPARK_SUBMIT_CLI = True
if IS_SPARK_SUBMIT_CLI:
    sc = SparkContext.getOrCreate()
    spark = SparkSession(sc)

# COMMAND ----------

if IS_SPARK_SUBMIT_CLI:
    coviddeath = spark.read.csv('UScasestemp1.csv', inferSchema=True, header=True)
else:
    coviddeath = spark.sql("SELECT * FROM uscasestemp1_csv")

# COMMAND ----------

data = coviddeath.select("Year","Date","Day", "Temp", "Admin2","Lat","Long","Province",((col("Case") > 2).cast("Double").alias("label")))
data = StringIndexer(inputCol='Admin2', outputCol='Admin2'+"_index").fit(data).transform(data)
data = StringIndexer(inputCol='Province', outputCol='Province'+"_index").fit(data).transform(data)

data.show(5)


# COMMAND ----------

splits = data.randomSplit([0.7, 0.3])
# for gradient boosted tree regression
dt_train = splits[0]
dt_test = splits[1].withColumnRenamed("label", "trueLabel")

print ("Training Rows:", dt_train.count(), " Testing Rows:", dt_test.count())

dt_train.show(5)


# COMMAND ----------

from pyspark.ml.classification import RandomForestClassifier
#assembler = VectorAssembler(inputCols =["Day","Temp","Lat","Long","Admin_index","Province_index"],outputCol="normfeatures")
assembler = VectorAssembler(inputCols =["Date","Day","Temp"],outputCol="normfeatures")
#assembler = VectorAssembler(inputCols =["Date","Year","Day","Temp"],outputCol="features")
minMax = MinMaxScaler(inputCol = assembler.getOutputCol(), outputCol="nfeatures")
featVect = VectorAssembler(inputCols=["nfeatures"], outputCol="features")
dt = RandomForestClassifier(labelCol="label", featuresCol="features",impurity="gini",featureSubsetStrategy="auto",numTrees=10,maxDepth=30,maxBins=128,seed=1234)
pipeline = Pipeline(stages=[assembler,minMax,featVect,dt])

piplineModel = pipeline.fit(dt_train)
print("Pipeline complete!") 
prediction = piplineModel.transform(dt_test)
predicted = prediction.select("features", "prediction", "trueLabel")
predicted.show(100, truncate=False)

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluation = MulticlassClassificationEvaluator(
    labelCol="trueLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluation.evaluate(prediction)
print(accuracy)

# COMMAND ----------

print("Test Error = %g" % (1.0 - accuracy))

#dt = DecisionTreeRegressor(featuresCol='features', labelCol='label', maxBins=77582)
#dt_pipeline = Pipeline(stages=[assembler, dt])
#paramGrid = ParamGridBuilder().build()
#cv = CrossValidator(estimator=dt_pipeline, evaluator=RegressionEvaluator(), estimatorParamMaps=paramGrid, numFolds=5)
#dt_model = cv.fit(dt_train)
#dt_prediction = dt_model.transform(dt_test)
#dt_predicted = dt_prediction.select("features", "prediction", "trueLabel")
#dt_predicted.show(10)

#dt_evaluator = RegressionEvaluator(labelCol="trueLabel", predictionCol="prediction", metricName="rmse")
#dt_rmse = dt_evaluator.evaluate(dt_prediction)


#print ("Root Mean Square Error (RMSE):", dt_rmse)
#from pyspark.ml.evaluation import MulticlassClassificationEvaluator
#evaluation = MulticlassClassificationEvaluator(
 #   labelCol="trueLabel", predictionCol="prediction", metricName="accuracy")
#accuracy = evaluation.evaluate(dt_prediction)
#print(accuracy)
