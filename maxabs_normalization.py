# -*- coding: utf-8 -*-

from __future__ import print_function

from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler
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
def maxAbsScalerModel(df, conf):
    """
        input: spark-dataFrame, conf [configuration params]
        return value: scaler, model
    """
    inp = conf.get("inputCol", None)
    output = conf.get("outputCol", None)
    scaler = MaxAbsScaler(inputCol = inp, outputCol = output)
    model = scaler.fit(df)
    return scaler, model

#transform data from fitted model into maximum absolute scaled model
def maxAbsTransformData(model, df):
    """
        input: maxAbsScalerModel, spark-dataFrame
        return value: scaled data frame
    """
    return model.transform(df)

#save maximum absolute scaler
def saveMaxAbsScaler(scaler, path):
    """
        input: scaler_model, path
        return value: None
    """
    scaler.save(path)

#save maximum absolute scaler
def saveMaxAbsScalerModel(model, path):
    """
        input: model, path
        return value: None
    """
    model.save(path)

#load maximum absolute scaler
def loadMaxAbsScaler(path):
    """
        input: path
        return value: scaler [MaxAbsScaler]
    """
    return MaxAbsScaler.load(path)

#load maximum absolute scaler model
def loadMaxAbsScalerModel(path):
    """
        input: path
        return value: model [MaxAbsScalerModel]
    """
    return MaxAbsScalerModel.load(path)

#save maximum absolute model (data frame)
def saveMaxAbsData(data, path, dataType):
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
def loadMaxAbsData(path):
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
    print(">>>=====(this is testing example)=====<<<")

    #create data frame        
    df = loadMaxAbsData("sample_maxabs_norm.csv")
    
    #assembling columns to vector
    dataFrame = convertToVector(df, ["col1", "col2", "col3"], "features")
        
    #create max absolute normalization scaler and model
    scaler, model = maxAbsScalerModel(dataFrame, config)

    #normalize data frame by using max absolute normalization
    data = maxAbsTransformData(model, dataFrame)

    #showing normalized data
    data.select("features", "scaledFeatures").show()

    #save data frame into desired path
    saveMaxAbsData(data, 'maxabs_norm_example.csv', 'csv')

    #save model into desired path
    saveMaxAbsScalerModel(model,'maxabs_norm_model')

    #load max absolute scaler model from desired path
    model2 = loadMaxAbsScalerModel('maxabs_norm_model')
    
    #transform data from loaded model
    data2 = maxAbsTransformData(model2, dataFrame)
    
    #showing normalized data
    data2.select("features", "scaledFeatures").show()
