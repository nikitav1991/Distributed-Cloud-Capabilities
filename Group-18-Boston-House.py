
from pyspark.sql.types import *
from pyspark.sql import Row
from pyspark import SparkContext
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from pyspark.mllib.util import MLUtils
from pyspark.mllib.feature import StandardScaler
from pyspark.ml.linalg import Vectors
from pyspark.sql import SQLContext
import pyspark.mllib
import pyspark.mllib.regression
from pyspark.mllib.regression import LabeledPoint
from pyspark.sql.functions import *
from pyspark.mllib.regression import LinearRegressionWithSGD

sc=SparkContext()
#import file
rdd = sc.textFile('tmp/boston/boston_house.csv')
#split comma seperated file based on ',' 
rdd = rdd.map(lambda line: line.split(","))
#column names in csv
header = rdd.first()
#remove column names
rdd = rdd.filter(lambda line:line != header)
#select all necessary columns in order
df = rdd.map(lambda line: Row(aamedv=line[13],crim = line[0], zn = line[1], indus=line[2], char=line[3], nox=line[4],
 rm=line[5], age=line[6], dis=line[7], rad=line[8], tax=line[9], ptratio=line[10], b=line[11], lstat=line[12]
 ))
#all columns except MEDV
features = df.map(lambda row: row[1:])
#use StandardScaler to scale training data values
standardizer = StandardScaler(withMean=True, withStd=True)
scaler = standardizer.fit(features)
features_transform = scaler.transform(features)
#create labels
lab = df.map(lambda row: row[0])
transformedData = lab.zip(features_transform)
transformedData = transformedData.map(lambda row: LabeledPoint(row[0],[row[1:]]))
#train the data using LinearRegressionWithSGD modeal
themodel= LinearRegressionWithSGD.train(transformedData,intercept=True)
#load test data csv file and repeat process till scaled values are obtained
rdd1 = sc.textFile('tmp/boston/verification.csv')
rdd1 = rdd1.map(lambda line: line.split(","))
header1 = rdd1.first()
rdd1 = rdd1.filter(lambda line:line != header1)
df1 = rdd1.map(lambda line: Row(crim = line[0], zn = line[1], indus=line[2], char=line[3], nox=line[4],
 rm=line[5], age=line[6], dis=line[7], rad=line[8], tax=line[9], ptratio=line[10], b=line[11], lstat=line[12]
 ))
features1 = df1.map(lambda row: row[0:])
features_transform1 = scaler.transform(features1)
#predict values using model and save predicted values
result = themodel.predict(features_transform1)
result.repartition(1).saveAsTextFile('tmp/boston/predicted3')

#Reference : http://www.techpoweredmath.com/spark-dataframes-mllib-tutorial/
