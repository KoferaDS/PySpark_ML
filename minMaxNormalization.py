# -*- coding: utf-8 -*-
"""
Created on Wed May 31 15:20:52 2018

@author: Ellen
"""
from __future__ import print_function
from pyspark.ml.linalg import Vectors

from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import MinMaxScalerModel

from pyspark.sql import SparkSession

#fit data frame into minimum maximum model
def scaleModel(dataFrame, conf):
    """
        input: dataFrame [spark.dataFrame], conf [configuration params]
        output: fitted model
    """
    input = conf["params"].get("inputCol")
    output = conf["params"].get("outputCol")
    minimum = conf["params"].get("min")
    maximum = conf["params"].get("max")
    scaler = MinMaxScaler(min = minimum, max = maximum, inputCol = input, 
                          outputCol = output)
    model = scaler.fit(dataFrame)
    return model

#transform fitted model into minimum maximum scaled model
def transformModel(dataFrame, conf):
    """
        input: dataFrame [spark.dataFrame], conf [configuration params]
        output: scaled data frame
    """
    model = scaleModel(dataFrame, conf)
    return model.transform(dataFrame)

#save minimum maximum scaler
def saveModel(conf, path):
    """
        input: configuration params for [MinMaxScaler], path
        output: void
    """
    input = conf["params"].get("inputCol")
    output = conf["params"].get("outputCol")
    minimum = conf["params"].get("min")
    maximum = conf["params"].get("max")
    scaler = MinMaxScaler(min = minimum, max = maximum, inputCol = input, 
                          outputCol = output)
    scaler.save(path)
    return

#load minimum maximum scaler
def loadModel(path):
    """
        input: path
        output: scaler [MinMaxScaler]
    """
    scaler = MinMaxScaler.load(path)
    return scaler

#save minimum maximum model (data frame)
def saveData(data, path, dataType):
    """
        input: data [data frame], path, data type (string)
        output: void
    """
    if (dataType == 'csv'):
        data.toPandas().to_csv(path)
    elif (dataType == 'html'):
        data.toPandas().to_html(path)
    elif (dataType == 'json'):
        data.toPandas().to_json(path)
    elif (dataType == 'pickle'):
        data.toPandas().to_pickle(path)
    elif (dataType == 'records'):
        data.toPandas().to_records(path)
    else:
        print("Setting defaults to csv")
        data.toPandas().to_csv(path)
    

#load minimum maximum model
def loadData(path):
    """
        input: path
        output: model [MinMaxScalerModel data frame]
    """
    model = MinMaxScalerModel.load(path)
    return model
    

#testing


spark = SparkSession\
    .builder\
    .appName("MinMaxScalerExample")\
    .getOrCreate()
    
dataFrame = spark.createDataFrame([
    (0, Vectors.dense([1.0, 0.1, -1.0]),),
    (1, Vectors.dense([2.0, 1.1, 1.0]),),
    (2, Vectors.dense([3.0, 10.1, 3.0]),)
], ["id", "features"])
  
config = {
        "params" : {
            "min" : 0.0,
            "max" : 1.0,
            "withStd" : True,
            "inputCol" : "features",
            "outputCol" : "scaledFeatures"
            }
        }
    
model = transformModel(dataFrame, config)

model.select("features", "scaledFeatures").show()
saveData(model, 'tes2.csv', 'csv')
#saveModel(config,'/tes')