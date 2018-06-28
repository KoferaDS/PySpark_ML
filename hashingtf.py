# -*- coding: utf-8 -*-

from __future__ import print_function

from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.ml.feature import HashingTF

sc = SparkContext.getOrCreate()
spark = SparkSession(sc)

def adaptParameter(parameter):
    '''
    Setting parameter from user input
        Input : - parameter from user
        Output : - parameters have been adjusted
    '''
    
    parameter_default = {
        "numFeatures" : 1 << 18,
        "binary" : False,
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

def transformData(df, param) :
    '''
    Transformed dataframe based on the formed model
        Input : - dataframe
                - formed model
        Output : - transformed dataframe
    '''
    
    hashingTF = HashingTF(numFeatures = param["numFeatures"],
                        inputCol = param["inputCol"],
                        outputCol = param["outputCol"])
    
    transform_df = hashingTF.transform(df)
    return transform_df
            
# ----------------Testing and Example--------------------#

if __name__ == "__main__" :
    
    df = spark.createDataFrame([(["a", "b", "c"],)], ["words"])
    
    conf = {
            "numFeatures" : 100, 
            "inputCol" : "words", 
            "outputCol" : "features"
            }
    
    new_conf = adaptParameter(conf)
    
    print(new_conf)
    
    transform_df = transformData(df, new_conf)
    
    print(transform_df)
    print(transform_df.show())
    print(transform_df.words)
    print(transform_df.features)
    
    df2 = spark.createDataFrame([(["a", "c", "c", "e", "e"],)], ["words"])
    
    conf2 = {
            "numFeatures" : 5, 
            "inputCol" : "words", 
            "outputCol" : "features"
            }
    
    new_conf2 = adaptParameter(conf2)
    
    transform_df2 = transformData(df2, new_conf2)
    
    print(transform_df2)
    print(transform_df2.words)
    print(transform_df2.features)
    
    spark.stop()