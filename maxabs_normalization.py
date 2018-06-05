# -*- coding: utf-8 -*-

from __future__ import print_function

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import MaxAbsScaler
from pyspark.ml.feature import MaxAbsScalerModel

from pyspark.sql import SparkSession

#configurations
  
spark = SparkSession\
    .builder\
    .appName("MaxAbsScalerExample")\
    .getOrCreate()
    
    
config = {
            "inputCol" : "features",
            "outputCol" : "scaledFeatures"    
        }

#fit data frame into maximum absolute model
def scaleModel(dataFrame,conf):
    """
        input: dataFrame [spark.dataFrame], conf [configuration params]
        output: fitted model
    """
    inp = conf.get("inputCol", None)
    output = conf.get("outputCol", None)
    scaler = MaxAbsScaler(inputCol = inp, outputCol = output)
    model = scaler.fit(dataFrame)
    return model

#transform data from fitted model into maximum absolute scaled model
def transformModel(dataFrame, conf):
    """
        input: dataFrame [spark.dataFrame], conf [configuration params]
        output: model [scaled data frame]
    """
    model = scaleModel(dataFrame, conf)
    return model.transform(dataFrame)

#save maximum absolute scaler
def saveModel(conf, path):
    """
        input: configuration params, path
        output: void
    """
    inp = conf.get("inputCol", None)
    output = conf.get("outputCol", None)
    scaler = MaxAbsScaler(inputCol = inp, outputCol = output)
    scaler.save(path)
    return

#load maximum absolute scaler
def loadModel(path):
    """
        input: path
        output: scaler [MaxAbsScaler]
    """
    scaler = MaxAbsScaler.load(path)
    return scaler

#save maximum absolute model (data frame)
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

#load maximum absolute model
def loadData(path):
    """
        input: path
        output: model [MaxAbsScalerModel data frame]
    """
    model = MaxAbsScalerModel.load(path)
    return model
    




#--------------------------Testing and Example--------------------------#

if __name__ == "__main__":

    #create data frame        
    dataFrame = spark.createDataFrame([
        (0, Vectors.dense([1.0, 0.1, -8.0]),),
        (1, Vectors.dense([2.0, 1.0, -4.0]),),
        (2, Vectors.dense([4.0, 10.0, 8.0]),)
    ], ["id", "features"])
        
    #normalize data frame by using min max normalization
    model = transformModel(dataFrame, config)

    #showting normalized data
    model.select("features", "scaledFeatures").show()

    #save data frame into desired path
    saveData(model, 'maxabs_norm_example.csv', 'csv')

    #save model into desired path
    saveModel(config,'maxabs_norm_model')