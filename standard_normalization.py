# -*- coding: utf-8 -*-

from __future__ import print_function
from pyspark.ml.linalg import Vectors

from pyspark.ml.feature import StandardScaler
from pyspark.ml.feature import StandardScalerModel

from pyspark.sql import SparkSession

#configurations
    
spark = SparkSession\
    .builder\
    .appName("StandardScalerExample")\
    .getOrCreate()
    
config = {
            "withMean" : False,
            "withStd" : True,
            "inputCol" : "features",
            "outputCol" : "scaledFeatures"
        }

#fit data frame into standard model
def standardScalerModel(df, conf):
    """
        input: spark-dataFrame, conf [configuration params]
        return value: model
    """
    mean = conf.get("withMean", False)
    std = conf.get("withStd", True)
    input = conf.get("inputCol", None)
    output = conf.get("outputCol", None)
    scaler = StandardScaler(inputCol = input, outputCol = output, 
                            withMean = mean, withStd = std)
    model = scaler.fit(dataFrame)
    return scaler, model

#transform fitted model into standard scaled model
def standardTransformData(model, dataFrame):
    """
        input: standardScalerModel, spark-dataFrame
        return value: scaled data frame
    """
    return model.transform(dataFrame)

#save standard scaler
def saveStandardScaler(scaler, path):
    """
        input: standardScaler, path
        return value: None
    """
    scaler.save(path)

#save standard scaler model
def saveStandardScalerModel(model, path):
    """
        input: standardScalerModel, path
        return value: None
    """
    model.save(path)

#load standard scaler
def loadStandardScaler(path):
    """
        input: path
        return value: scaler [StandardScaler]
    """
    return StandardScaler.load(path)

#load standard scaler model
def loadStandardScalerModel(path):
    """
        input: path
        return value: model [StandardScalerModel]
    """
    return StandardScalerModel.load(path)

#save standard model (data frame)
def saveStandardData(data, path, dataType):
    """
        input: data [data frame], path, data type (string)
        return value: void
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

#load data frame
def loadStandardData(path):
    """
        input: path
        output: df [data frame]
    """
    df = spark.read.format("libsvm").load(path)
    return df




#--------------------------Testing and Example--------------------------#

if __name__ == "__main__":

    #create data frame        
    dataFrame = spark.createDataFrame([
        (0, Vectors.dense([1.0, 0.1, -8.0]),),
        (1, Vectors.dense([2.0, 1.0, -4.0]),),
        (2, Vectors.dense([4.0, 10.0, 8.0]),)
    ], ["id", "features"])
        
    #create standard normalization scaler and model
    scaler, model = standardScalerModel(dataFrame, config)

    #normalize data frame by using standard normalization
    data = standardTransformData(model, dataFrame)

    #showing normalized data
    data.select("features", "scaledFeatures").show()

    #save data frame into desired path
    saveStandardData(data, 'standard_norm_example.csv', 'csv')

    #save model into desired path
    saveStandardScalerModel(model, 'standard_norm_model')

    #load standard scaler model from desired path
    model2 = loadStandardScalerModel('standard_norm_model')

    #transform data from loaded model
    data2 = standardTransformData(model2, dataFrame)

    #showing normalized data
    data2.select("features", "scaledFeatures").show()
