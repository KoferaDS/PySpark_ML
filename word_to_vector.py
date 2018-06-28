# -*- coding: utf-8 -*-

from __future__ import print_function

from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.ml.feature import Word2VecModel, Word2Vec

sc = SparkContext.getOrCreate()
spark = SparkSession(sc)

def adaptParameter(parameter):
    '''
    Setting parameter from user input
        Input : - parameter from user
        Output : - parameters have been adjusted
    '''
    
    parameter_default = {
        "vectorSize" : 100, 
        "stepSize" : 0.025,
        "numPartitions" : 1,
        "maxIter" : 1,
        "minCount" : 5, 
        "seed" : None,
        "inputCol" : None, 
        "outputCol" : None,
        "windowSize" : 5,
        "maxSentenceLength" : 1000
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

def train(df, param) :
    '''
    Word2Vec training, returning Word2Vec model.
        Input : - Dataframe
                - parameter
        Output : - Word2Vec model
    '''
    
    word2vec = Word2Vec(vectorSize = param["vectorSize"],
                        stepSize = param["stepSize"],
                        numPartitions = param["numPartitions"],
                        maxIter = param["maxIter"],
                        minCount = param["maxIter"],
                        seed = param["seed"],
                        inputCol = param["inputCol"],
                        outputCol = param["outputCol"],
                        windowSize = param["windowSize"],
                        maxSentenceLength = param["maxSentenceLength"]
                        )
            
    model = word2vec.fit(df)
    return model
 
def saveModel(model,path):
    '''
    Save model into corresponding path
        Input : - model
                - path
        Output : - saved model
    '''
    
    model.save(path)

def loadModel(path):
    '''
    Load Word2Vec model
        Input : - path
        Output : - model [Word2Vec model data frame]
    '''
    
    model = Word2VecModel.load(path)
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

def foundWord(word, sent) :
    '''
    Find word in sentences
        Input : - word
                - sentences
        Output : - flag (True/False)
    '''
    flag = False
    
    for i in sent :
        if word in i :
            flag = True
            
    return flag

def getSynonyms(word, sent, num, model) :
    '''
        Input : - word
                - sentences
                - number
                - fitted model
        Output : - dataframe with two fields word and similarity
    '''
    
    temp = ''
    flag = foundWord(word, sent)
    
    if (flag) :
        temp = model.findSynonyms(word, num).select("word", 
                                 "similarity").show()
    else :
        temp = 'Not working'
          
    return temp

def loadWord2VecData(path):
    '''
    Load data frame
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
    elif (path.lower().find(".parquet") != -1) :
        df = spark.read.parquet(path)    
        sentences = df.take(1)[0].text
    elif (path.lower().find(".txt") != -1) :
        df = spark.read.text(path)
        sentences = df.take(1)[0].value
    else :
        print("Unsupported yet ...")

    return sentences
            
# ----------------Testing and Example--------------------#
    
if __name__ == "__main__" :
        
    df_txt = loadWord2VecData("D:/elephant.txt")
    
    load_text = (df_txt).split(" ")
    
    for i in range (len(load_text)) :
        if load_text[i][len(load_text[i])-1] == '.' :
            load_text[i] = load_text[i][:len(load_text[i])-1]
             
    doc = spark.createDataFrame([(load_text,)], ["sentence"])
    
    doc.show()
    
    conf = {
            "vectorSize" : 5,
            "seed" : 42,
            "inputCol": "sentence",
            "outputCol": "model"
            }
    
    new_conf = adaptParameter(conf)
    
    model = train(doc, new_conf)
    model.getVectors().show()
    
    if (foundWord("elephant", load_text)) :
        getSynonyms("elephant", load_text, 2, model)
    else :
        print(getSynonyms("elephant", load_text, 2, model))
        
    spark.stop()
    