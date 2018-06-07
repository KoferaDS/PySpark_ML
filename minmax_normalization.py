# -*- coding: utf-8 -*-

from __future__ import print_function
from pyspark.ml.linalg import Vectors

from pyspark.ml.feature import MinMaxScaler
from pyspark.ml.feature import MinMaxScalerModel

from pyspark.sql import SparkSession



#configurations

spark = SparkSession\
    .builder\
    .appName("MinMaxScalerExample")\
    .getOrCreate()
    
#config input example  
config = {
            "min" : 0.0,
            "max" : 1.0,
            "withStd" : True,
            "inputCol" : "features",
            "outputCol" : "scaledFeatures"
           }


def minMaxScalerModel(df, conf):
    """
    input : spark-dataframe, conf
    return value : model
    """
    
    input = conf.get("inputCol", None)
    output = conf.get("outputCol", None)
    minimum = conf.get("min", 0.0)
    maximum = conf.get("max", 1.0)
    scaler = MinMaxScaler(min = minimum, max = maximum, inputCol = input, 
                          outputCol = output)
    return scaler.fit(df)



#transform fitted model into minimum maximum scaled model
def minMaxTransformData(model, df):
    """
        input : MinMaxScalerModel, spark-dataframe
        return value : scaled data frame
    """
    return model.transform(df)

#save minimum maximum scaler
def saveMinMaxScaler(scaler, path):
    """
    this function used for save the Scaler
        input: scaler_model, path
        return value: None
    """
    scaler.save(path)
   
#save model (fitted scaler)
def saveMinMaxScalerModel(model, path):
    """
        input: model, path
        return value: None
    """    
    model.save(path)

#load the MinMax Scaler
def loadMinMaxScaler(path):
    """
        input: path
        output: scaler [MinMaxScaler]
    """
    scaler = MinMaxScaler.load(path)
    return scaler   
 
#load minimum MinMaxScaler Model
def loadMinMaxScalerModel(path):
    """
        input: path
        output: scaler [MinMaxScaler]
    """
    scaler = MinMaxScalerModel.load(path)
    return scaler


#save minimum maximum model (data frame)
def saveMinMaxData(data, path, dataType):
    """
        input: data [spark-dataframe], path, data type (string)
        output: None
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
def loadMinMaxData(path):
    """
        input: path
        output: df [data frame]
    """
    df = spark.read.format("libsvm").load(path)
    return df
    


#--------------------------Testing and Example--------------------------#

if __name__ == "__main__":
    print (">>>=====(this is testing example)=====<<<")

    #create data frame        
    dataFrame = spark.createDataFrame([
        (0, Vectors.dense([1.0, 0.1, -1.0]),),
        (1, Vectors.dense([2.0, 1.1, 1.0]),),
        (2, Vectors.dense([3.0, 10.1, 3.0]),)
    ], ["id", "features"])
        
    #create min max normalization model
    model = minMaxScalerModel(dataFrame, config)
    
    #normalize data frame by using min max normalization
    data = minMaxTransformData(model, dataFrame)

    #showting normalized data
    data.select("features", "scaledFeatures").show()

    #save data frame into desired path
    saveMinMaxData(model, 'minmax_norm_example.csv', 'csv')

    #save model into desired path
    saveMinMaxScalerModel(model, 'minmax_norm_model')

    #load min max scaler model from desired path
    model2 = loadMinMaxScalerModel('minmax_norm_model')

    #transform data from loaded model
    data2 = minMaxTransformData(model2, dataFrame)

    #showing normalized data
    data2.select("features", "scaledFeatures").show()
