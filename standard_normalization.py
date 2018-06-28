# -*- coding: utf-8 -*-

from __future__ import print_function
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

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
        data.toPandas().to_csv(path, float_format = None)
    elif (dataType == 'html'):
        data.toPandas().to_html(path)
    elif (dataType == 'json'):
        data.toPandas().to_json(path)
    elif (dataType == 'pickle'):
        data.toPandas().to_pickle(path)
    elif (dataType == 'records'):
        data.toPandas().to_records(path)
    elif (dataType == 'txt'):
        data.toPandas().to_csv(path, sep = '\t', index = False, header = False)
    else:
        print("Setting defaults to csv")
        data.toPandas().to_csv(path)

#load data frame
def loadStandardData(path):
    """
        input: path
        output: df [data frame]
    """
#    if (path.lower().find(".txt") != -1):
#        df = spark.read.format("libsvm").load(path, header = False, inferSchema = "true")
    if (path.lower().find(".csv") != -1):
        df = spark.read.format("csv").load(path, header = True, inferSchema = "true")
    elif (path.lower().find(".json") != -1):
        df = spark.read.json(path, header = True, inferSchema = "true")
    elif (path.lower().find(".md") != -1):
        df = spark.read.textFile(path, header = True, inferSchema = "true")
    else:
        print("Unsupported yet.")

    return df

    

#--------------------------Testing and Example--------------------------#

if __name__ == "__main__":

    #create data frame        
    df = loadStandardData("standard_norm_sample.csv")
        
    #assembling columns to vector
    assembler = VectorAssembler(
        inputCols=["col1", "col2", "col3"],
        outputCol="features")
    
    dataFrame = assembler.transform(df)
    
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