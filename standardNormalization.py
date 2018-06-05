# -*- coding: utf-8 -*-
"""
Created on Thu May 31 10:31:37 2018

@author: Ellen
"""
from __future__ import print_function
from pyspark.ml.linalg import Vectors

from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import StandardScalerModel

from pyspark.sql import SparkSession

#fit data frame into standard model
def scaleModel(dataFrame,conf):
    """
        input: dataFrame [spark.dataFrame], conf [configuration params]
        output: fitted model
    """
    mean = conf["params"].get("withMean")
    std = conf["params"].get("withStd")
    input = conf["params"].get("inputCol")
    output = conf["params"].get("outputCol")
    scaler = StandardScaler(inputCol = input, outputCol = output, 
                            withMean = mean, withStd = std)
    model = scaler.fit(dataFrame)
    return model

#transform fitted model into standard scaled model
def transformModel(dataFrame, conf):
    """
        input: dataFrame [spark.dataFrame], conf [configuration params]
        output: scaled data frame
    """
    model = scaleModel(dataFrame, conf)
    transformed = model.transform(dataFrame)
    return transformed

#save standard scaler
def saveModel(conf, path):
    """
        input: configuration params, path
        output: void
    """
    mean = conf["params"].get("withMean")
    std = conf["params"].get("withStd")
    input = conf["params"].get("inputCol")
    output = conf["params"].get("outputCol")
    scaler = StandardScaler(inputCol = input, outputCol = output, 
                            withMean = mean, withStd = std)
    scaler.save(path)
    return

#load standard scaler
def loadModel(path):
    """
        input: path
        output: scaler [StandardScaler]
    """
    scaler = StandardScaler.load(path)
    return scaler

#save standard model (data frame)
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

#load standard model
def loadData(path):
    """
        input: path
        output: model [StandardScalerModel data frame]
    """
    model = StandardScalerModel.load(path)
    return model



#testing
    

    
spark = SparkSession\
    .builder\
    .appName("StandardScalerExample")\
    .getOrCreate()
    
config = {
        "params" : {
                "withMean" : False,
                "withStd" : True,
                "inputCol" : "features",
                "outputCol" : "scaledFeatures"
                }
        }
        
dataFrame = spark.createDataFrame([
    (0, Vectors.dense([1.0, 0.1, -8.0]),),
    (1, Vectors.dense([2.0, 1.0, -4.0]),),
    (2, Vectors.dense([4.0, 10.0, 8.0]),)
], ["id", "features"])
    
model = transformModel(dataFrame, config)
model.select("features", "scaledFeatures").show()
