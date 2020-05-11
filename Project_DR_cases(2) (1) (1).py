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

data = coviddeath.select("Year","Date","Day", "Temp", "Admin2","Lat","Long","Province",col("Case").alias("label"))
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


#assembler = VectorAssembler(inputCols =["Day","Temp","Lat","Long","Admin_index","Province_index"],outputCol="features")
assembler = VectorAssembler(inputCols =["Date","Year","Day","Temp","Admin2_index"],outputCol="features")
#assembler = VectorAssembler(inputCols =["Date","Year","Day","Temp"],outputCol="features")

dt = DecisionTreeRegressor(featuresCol='features', labelCol='label', maxBins=77582)
dt_pipeline = Pipeline(stages=[assembler, dt])
paramGrid = ParamGridBuilder().build()
cv = CrossValidator(estimator=dt_pipeline, evaluator=RegressionEvaluator(), estimatorParamMaps=paramGrid, numFolds=5)
dt_model = cv.fit(dt_train)
dt_prediction = dt_model.transform(dt_test)
dt_predicted = dt_prediction.select("features", "prediction", "trueLabel")
dt_predicted.show(10)

dt_evaluator = RegressionEvaluator(labelCol="trueLabel", predictionCol="prediction", metricName="rmse")
dt_rmse = dt_evaluator.evaluate(dt_prediction)


print ("Root Mean Square Error (RMSE):", dt_rmse)
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluation = MulticlassClassificationEvaluator(
    labelCol="trueLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluation.evaluate(dt_prediction)
print(accuracy)
