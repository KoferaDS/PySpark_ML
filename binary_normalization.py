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

#create binary normalizer model
def binaryScalerModel(df, conf):
    """
        input: spark-dataFrame, conf [configuration params]
        return value: model
    """
    input = conf.get("inputCol", None)
    output = conf.get("outputCol", None)
    tres = conf.get("threshold", 0.0)
    model = Binarizer(threshold = tres,inputCol = input, outputCol = output)
    return model

#transform data from unfitted model into binary form
def binaryTransformData(model, df):
    """
        input: binaryScalerModel, spark-dataFrame
        return value: scaled data frame
    """
    return model.transform(dataFrame)

#save binary scaler
def saveBinaryScaler(scaler, path):
    """
        input: binaryScalerModel, path
        return value: None
    """
    scaler.save(path)

#load binary scaler
def loadBinaryScaler(path):
    """
        input: path
        return value: binaryScalerModel
    """
    return Binarizer.load(path)

#save binary model (data frame)
def saveBinaryData(data, path, dataType):
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
def loadBinaryData(path):
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
        (0, 0.1),
        (1, 0.8),
        (2, 0.2)
    ], ["id", "features"])
        
    #create binary scaler model
    model = binaryScalerModel(dataFrame, config)

    #normalize data frame by using binary normalization
    data = binaryTransformData(model, dataFrame)

    #showting normalized data
    data.show()

    #save data frame into desired path
    saveBinaryData(data, 'binary_norm_example.csv', 'csv')

    #save model into desired path
    saveBinaryScaler(model,'binary_norm_model')

    #load binary scaler model from desired path
    model2 = loadBinaryScaler('binary_norm_model')

    #transform data from loaded model
    data2 = binaryTransformData(model2, dataFrame)

    #showing normalized data
    data2.select("features", "binarizedFeatures").show()
