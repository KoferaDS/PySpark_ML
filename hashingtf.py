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

def loadHashingTFData(path):
    '''
    Load list of words
        Input : - path
        Output : - data frame
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
    
    data = loadHashingTFData("D:/elephant.txt")
    
    print(data)
    
    df = spark.createDataFrame([(data,)], ["words"])
    
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
    
    '''
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
    '''
    
    spark.stop()