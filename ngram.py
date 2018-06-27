# -*- coding: utf-8 -*-

from __future__ import print_function

from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.ml.feature import NGram
from pyspark.sql import Row

sc = SparkContext.getOrCreate()
spark = SparkSession(sc)

def adaptParameter(parameter):
    '''
    Setting parameter from user input
        Input : - parameter from user
        Output : - parameters have been adjusted
    '''
    
    parameter_default = {
        "n" : 2, 
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

def transformData(df, parameter) :
    '''
    Transformed dataframe based on the parameter
        Input : - parameter
        Output : - transformed dataframe
    '''
    ngram = NGram(n= parameter["n"], 
                  inputCol = parameter["inputCol"],
                  outputCol = parameter["outputCol"])
    
    temp = ''
    if len(row.inputTokens) < ngram.getN() :
        temp = 'No element in ' + parameter["outputCol"]
    else :
        temp  = ngram.transform(df).head()
    
    return temp

# ----------------Testing and Example--------------------#
    
if __name__ == "__main__" :
    
    row = Row(inputTokens=["a", "b", "c", "d", "e"])
    df = spark.createDataFrame([row])
    
    conf = {
            "n" : 4, 
            "inputCol" : "inputTokens", 
            "outputCol" : "nGrams"
            }
    
    new_conf = adaptParameter(conf)
    
    transform_df = transformData(df, new_conf)
    print(transform_df)
    
    row2 = Row(inputTokens=["a", "a", "c", "c", "e", "e"])
    df2 = spark.createDataFrame([row2])
    
    conf2 = {
            "n" : 10, 
            "inputCol" : "inputTokens", 
            "outputCol" : "nGrams"
            }
    
    new_conf2 = adaptParameter(conf2)
    
    transform_df2 = transformData(df2, new_conf2)
    print(transform_df2)

    spark.stop()