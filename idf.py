# -*- coding: utf-8 -*-

from __future__ import print_function

from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.ml.feature import IDF, IDFModel
from pyspark.ml.linalg import DenseVector

sc = SparkContext.getOrCreate()
spark = SparkSession(sc)

def adaptParameter(parameter):
    '''
    Setting parameter from user input
        Input : - parameter from user
        Output : - parameters have been adjusted
    '''
    
    parameter_default = {
        "minDocFreq" : 0,
        "inputCol" : None, 
        "outputCol" : None
        }
    
    param_keys = list(parameter.keys())
    paramDefault_keys = list(parameter_default.keys())
    
    new_parameter = {}
    
    for par in paramDefault_keys :
        if par in param_keys :
            new_parameter[par] = parameter[par]
        else :
            new_parameter[par] = parameter_default[par]
            
    return new_parameter

def train(df, parameter) :
    '''
    IDF training, returning IDF model.
    input : - Dataframe
           - parameter
    
    output : IDF model
    '''
    
    idf = IDF(minDocFreq = parameter["minDocFreq"], 
              inputCol = parameter["inputCol"],
              outputCol = parameter["outputCol"])
            
    model = idf.fit(df)
    return model
    
def saveModel(model,path):
    '''
    Save model into corresponding path
        Input : - model
                - path
            Output : -saved model
    '''
    
    model.save(path)

def loadModel(path):
    '''
    Load IDFModel
        input : - path
        output: - model [IDFModel data frame]
    '''
    
    model = IDFModel.load(path)
    return model

def copyModel(model):
    '''
    Returning copied model
        Input: - model
        Output: - model of copy
    '''
    
    copy_model = model.copy(extra=None)
    return copy_model

def transformData(df, model):
    '''
    Transformed dataframe based on the formed model
        Input : - dataframe
                - formed model
        Output : - transformed dataframe
    
    '''
    
    return model.transform(df)

# ----------------Testing and Example--------------------#
    
if __name__ == "__main__" :
        
    df = spark.createDataFrame([(DenseVector([1.0, 2.0]),),
                                (DenseVector([0.0, 1.0]),), 
                                (DenseVector([3.0, 0.2]),)], ["tf"])
    
    conf = {
            "minDocFreq" : 3,
            "inputCol" : "tf",
            "outputCol" : "idf"
            }
    
    new_conf = adaptParameter(conf)
    
    model = train(df, new_conf)
    print(transformData(df, model).head())
    
    spark.stop()