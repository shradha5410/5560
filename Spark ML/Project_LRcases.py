# Databricks notebook source
# Import Spark SQL and Spark ML libraries
from pyspark.sql.types import *
from pyspark.sql.functions import *
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.ml.evaluation import BinaryClassificationEvaluator, RegressionEvaluator
from pyspark.mllib.tree import RandomForest
from pyspark.ml import Pipeline

from pyspark.ml.classification import DecisionTreeClassifier

from pyspark.ml.feature import VectorAssembler, StringIndexer, VectorIndexer, MinMaxScaler
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

from pyspark.ml.classification import LogisticRegression
from pyspark.ml.feature import VectorAssembler

from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.classification import DecisionTreeClassifier, RandomForestClassifier



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

data = coviddeath.select("Year","Date","Day", "Temp","Lat","Long","Admin2","Province",((col("Case") > 2).cast("Double").alias("label")))
data = StringIndexer(inputCol='Admin2', outputCol='Admin2'+"_index").fit(data).transform(data)
data = StringIndexer(inputCol='Province', outputCol='Province'+"_index").fit(data).transform(data)
data.show(5)


# COMMAND ----------

splits = data.randomSplit([0.7, 0.3])
train = splits[0]
test = splits[1].withColumnRenamed("label", "trueLabel")
train_rows = train.count()
test_rows = test.count()
print ("Training Rows:", train_rows, " Testing Rows:", test_rows)


# COMMAND ----------

from pyspark.ml.classification import RandomForestClassifier
assembler = VectorAssembler(inputCols =["Day","Temp","Lat","Province_index","Admin2_index"],outputCol="normfeatures")
minMax = MinMaxScaler(inputCol = assembler.getOutputCol(), outputCol="nfeatures")
featVect = VectorAssembler(inputCols=["nfeatures"], outputCol="features")
lr = LogisticRegression(labelCol="label",featuresCol="features",maxIter=10,regParam=0.3)
pipeline = Pipeline(stages=[assembler,minMax,featVect,lr])

# COMMAND ----------

piplineModel = pipeline.fit(train)
print("Pipeline complete!")


# COMMAND ----------

# piplineModel with train data set applies test data set and generate predictions
prediction = piplineModel.transform(test)
predicted = prediction.select("features", "prediction", "trueLabel")
predicted.show(100, truncate=False)

# COMMAND ----------

from pyspark.ml.evaluation import MulticlassClassificationEvaluator
evaluation = MulticlassClassificationEvaluator(
    labelCol="trueLabel", predictionCol="prediction", metricName="accuracy")
accuracy = evaluation.evaluate(prediction)
print("Accuracy of Logistic Regression is: ",accuracy)

# COMMAND ----------

print("Test Error = %g" % (1.0 - accuracy))

# COMMAND ----------


