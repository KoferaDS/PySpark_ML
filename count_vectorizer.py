# -*- coding: utf-8 -*-

from __future__ import print_function

from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.ml.feature import CountVectorizer, CountVectorizerModel

sc = SparkContext.getOrCreate()
spark = SparkSession(sc)

def adaptParameter(parameter):
    '''
    Setting parameter from user input
        Input : - parameter from user
        Output : - parameters have been adjusted
    '''
    
    parameter_default = {
        "inputCol" : None, 
        "outputCol" : None,
        "minTF" : 1.0,
        "minDF" : 1.0,
        "vocabSize" : 1 << 18,
        "binary" : False
        }
    
    param_keys = list(parameter.keys())
    paramDefault_keys = list(parameter_default.keys())
    
    new_parameter = {}
    
    for par in paramDefault_keys :
        if par not in param_keys :
            new_parameter[par] = parameter_default[par]
        else :
            new_parameter[par] = parameter[par]
            
    return new_parameter


def train(df, param) :
    '''
    CountVectorizer training, returning CountVectorizer model.
    input : - Dataframe
           - parameter
    
    output : CountVectorizer model
    '''
    
    cv = CountVectorizer(inputCol = param["inputCol"],
                        outputCol = param["outputCol"],
                        minTF = param["minTF"],
                        minDF = param["minDF"],
                        vocabSize = param["vocabSize"],
                        binary = param["binary"])
            
    model = cv.fit(df)
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
    Load Count Vectorizer model
        input : - path
        output: - model [Count Vectorizer model data frame]
    '''
    
    model = CountVectorizerModel.load(path)
    return model

def copyModel(model):
    '''
    Returning copied model
    Input: model
    Output: model of copy
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

def getVocab(model) :
    '''
    Get array of terms in the vocabulary.
        Input : - fitted model
        Output : - array of terms
    '''
    return model.vocabulary

# ----------------Testing and Example--------------------#
    
if __name__ == "__main__" :
    
    #Input data: Each row is a bag of words with a ID.
    df = spark.createDataFrame([
       (0, "a b c".split(" ")),
       (1, "a b b c a".split(" "))
    ], ["id", "words"])
    
    conf = {
            "inputCol" : "words", 
            "outputCol" : "features", 
            "vocabSize" : 3, 
            "minDF" : 2.0
            }
    
    new_conf = adaptParameter(conf)
    
    print(new_conf)
    
    #fit a CountVectorizerModel from the corpus.
    model = train(df, new_conf)
    print(model)
    
    result = transformData(df, model)
    
    print(result)
    result.show(truncate=False)
    
    print(getVocab(model))
    
    #Input data: Each row is a bag of words with a ID.
    #df2 = spark.createDataFrame([
    #   (0, "a b c e f".split(" ")),
    #   (1, "a b b c a d d".split(" "))
    #], ["id", "words"])
    
    #conf2 = {
    #        "inputCol" : "words", 
    #        "outputCol" : "features", 
    #        "vocabSize" : 6, 
    #        "minTF" : 2.0
    #        }
    
    #new_conf2 = adaptParameter(conf2)
    
    #fit a CountVectorizerModel from the corpus.
    #model2 = train(df2, new_conf2)
    #result2 = transformData(df2, model2)
    #result2.show(truncate=False)
    
    #print(getVocab(model2))
    
    spark.stop()