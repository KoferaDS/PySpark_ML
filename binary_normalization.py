# -*- coding: utf-8 -*-

from __future__ import print_function

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import Binarizer

from pyspark.sql import SparkSession

#configurations

spark = SparkSession\
    .builder\
    .appName("StandardScalerExample")\
    .getOrCreate()
    
config = {
            "threshold" : 0.5,
            "inputCol" : "features",
            "outputCol" : "binarizedFeatures"  
        }

#transform data from unfitted model into binary form
def transformModel(dataFrame, conf):
    """
        input: dataFrame [spark.dataFrame], conf [configuration params]
        output: binary normalized data frame
    """
    input = conf.get("inputCol")
    output = conf.get("outputCol")
    tres = conf.get("threshold")
    scaler = Binarizer(threshold = tres,inputCol = input, outputCol = output)
    model = scaler.transform(dataFrame)
    return model

#save binary scaler
def saveModel(conf,path):
    """
        input: configuration params, path
        output: void
    """
    input = conf.get("inputCol")
    output = conf.get("outputCol")
    tres = conf.get("threshold")
    scaler = Binarizer(threshold = tres,inputCol = input, outputCol = output)
    scaler.save(path)

#load binary scaler
def loadModel(path):
    """
        input: path
        output: scaler [Binarizer]
    """
    scaler = Binarizer.load(path)
    return scaler

#save binary model (data frame)
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




#--------------------------Testing and Example--------------------------#

if __name__ == "__main__":

    #create data frame        
    dataFrame = spark.createDataFrame([
        (0, 0.1),
        (1, 0.8),
        (2, 0.2)
    ], ["id", "features"])
        
    #normalize data frame by using min max normalization
    model = transformModel(dataFrame, config)

    #showting normalized data
    model.show()

    #save data frame into desired path
    saveData(model, 'binary_norm_example.csv', 'csv')

    #save model into desired path
    saveModel(config,'binary_norm_model')