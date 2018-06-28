# -*- coding: utf-8 -*-

from __future__ import print_function
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
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
    model =  scaler.fit(df)
    return scaler, model

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
        return value: scaler [MinMaxScaler]
    """
    scaler = MinMaxScaler.load(path)
    return scaler   
 
#load minimum MinMaxScaler Model
def loadMinMaxScalerModel(path):
    """
        input: path
        return value: scaler [MinMaxScaler]
    """
    scaler = MinMaxScalerModel.load(path)
    return scaler


#save minimum maximum model (data frame)
def saveMinMaxData(data, path, dataType):
    """
        input: data [spark-dataframe], path, data type (string)
        return value: None
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
        return value: df [data frame]
    """
    if (path.lower().find(".csv") != -1):
        df = spark.read.format("csv").load(path, header = True, inferSchema = "true")
    elif (path.lower().find(".json") != -1):
        df = spark.read.json(path, header = True, inferSchema = "true")
    elif (path.lower().find(".md") != -1):
        df = spark.read.textFile(path, header = True, inferSchema = "true")
    else:
        print("Unsupported yet.")

    return df
    
#convert column(s) into vector dense
def convertToVector(df, inCol, outCol):
    """
        input: df [spark-dataFrame], inCol [list], outCol [string]
        return value: df [spark-dataFrame]
    """
    if (outCol == None):
        outCol = "features"
    assembler = VectorAssembler(
        inputCols=inCol,
        outputCol=outCol)
    
    return assembler.transform(df)


#--------------------------Testing and Example--------------------------#

if __name__ == "__main__":
    print (">>>=====(this is testing example)=====<<<")

    #create data frame        
    df = loadMinMaxData("sample_minmax_norm.csv")
    
    dataFrame = convertToVector(df, ["col1", "col2", "col3"], "features")    
    
    #create min max normalization scaler and model
    scaler, model = minMaxScalerModel(dataFrame, config)
    
    #normalize data frame by using min max normalization
    data = minMaxTransformData(model, dataFrame)

    #showting normalized data
    data.select("features", "scaledFeatures").show()

    #save data frame into desired path
    saveMinMaxData(data, 'minmax_norm_example.csv', 'csv')

    #save model into desired path
    saveMinMaxScalerModel(model, 'minmax_norm_model')

    #load min max scaler model from desired path
    model2 = loadMinMaxScalerModel('minmax_norm_model')

    #transform data from loaded model
    data2 = minMaxTransformData(model2, dataFrame)

    #showing normalized data
    data2.select("features", "scaledFeatures").show()
