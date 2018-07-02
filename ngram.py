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
    
    if len(ngram.transform(df).head().inputTokens) < ngram.getN() :
        print('No element in ' + parameter["outputCol"])
    else :
        temp  = ngram.transform(df).show()
    
    return temp

def loadNGramData(path):
    '''
    Load list of words
        Input : - path
        Output : - list of words
    '''
    
    if (path.lower().find(".csv") != -1) :
        df = spark.read.load(path,
                     format="csv", sep=":", inferSchema="true", header="true")
        sentences = df.take(1)[0].text
    elif (path.lower().find(".json") != -1) :
        df = spark.read.json(path)
        sentences = df.take(1)[0].text
    elif (path.lower().find(".txt") != -1) :
        df = spark.read.text(path)
        sentences = df.take(1)[0].value
    else :
        print("Unsupported yet ...")
        
    split_sent = (sentences).split(" ")
    
    for i in range (len(split_sent)) :
        if split_sent[i][len(split_sent[i])-1] == '.' :
            split_sent[i] = split_sent[i][:len(split_sent[i])-1]

    return split_sent

# ----------------Testing and Example--------------------#
    
if __name__ == "__main__" :
    
    load_text = loadNGramData("D:/elephant.txt")
    
    print(load_text)
            
    row = Row(inputTokens=load_text)
    df = spark.createDataFrame([row])
    
    conf = {
            "n" : 2, 
            "inputCol" : "inputTokens", 
            "outputCol" : "nGrams"
            }
    
    new_conf = adaptParameter(conf)

    transform_df = transformData(df, new_conf)
    
    row2 = Row(inputTokens=["a", "a", "c", "c", "e", "e"])
    df2 = spark.createDataFrame([row2])
    
    conf2 = {
            "n" : 3, 
            "inputCol" : "inputTokens", 
            "outputCol" : "nGrams"
            }
    
    new_conf2 = adaptParameter(conf2)
    
    transform_df2 = transformData(df2, new_conf2)

    spark.stop()