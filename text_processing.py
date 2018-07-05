# -*- coding: utf-8 -*-

from __future__ import print_function

from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.ml.feature import NGram, CountVectorizer, Word2Vec, IDF, HashingTF
from pyspark.ml.feature import Word2VecModel, CountVectorizerModel, IDFModel
from pyspark.sql import Row
from pyspark.ml.linalg import DenseVector

sc = SparkContext.getOrCreate()
spark = SparkSession(sc)

def loadData(path):
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

def word2VecTrain(df, param) :
    '''
    Word2Vec training, returning Word2Vec model.
        Input : - Dataframe
                - parameter
        Output : - Word2Vec model
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
    
    param_keys = list(param.keys())
    paramDefault_keys = list(parameter_default.keys())
    
    new_parameter = {}
    
    for par in paramDefault_keys :
        if par in param_keys :
            new_parameter[par] = param[par]
        else :
            new_parameter[par] = parameter_default[par]
    
    word2vec = Word2Vec(vectorSize = new_parameter["vectorSize"],
                        stepSize = new_parameter["stepSize"],
                        numPartitions = new_parameter["numPartitions"],
                        maxIter = new_parameter["maxIter"],
                        minCount = new_parameter["maxIter"],
                        seed = new_parameter["seed"],
                        inputCol = new_parameter["inputCol"],
                        outputCol = new_parameter["outputCol"],
                        windowSize = new_parameter["windowSize"],
                        maxSentenceLength = new_parameter["maxSentenceLength"]
                        )
            
    model = word2vec.fit(df)
    return model

def idfTrain(df, parameter) :
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

def countVectorizerTrain(df, param) :
    '''
    CountVectorizer training, returning CountVectorizer model.
    input : - Dataframe
           - parameter
    
    output : CountVectorizer model
    '''
    
    parameter_default = {
        "inputCol" : None, 
        "outputCol" : None,
        "minTF" : 1.0,
        "minDF" : 1.0,
        "vocabSize" : 1 << 18,
        "binary" : False
        }
    
    param_keys = list(param.keys())
    paramDefault_keys = list(parameter_default.keys())
    
    new_parameter = {}
    
    for par in paramDefault_keys :
        if par not in param_keys :
            new_parameter[par] = parameter_default[par]
        else :
            new_parameter[par] = param[par]
    
    cv = CountVectorizer(inputCol = new_parameter["inputCol"],
                        outputCol = new_parameter["outputCol"],
                        minTF = new_parameter["minTF"],
                        minDF = new_parameter["minDF"],
                        vocabSize = new_parameter["vocabSize"],
                        binary = new_parameter["binary"])
            
    model = cv.fit(df)
    return model

def transformNGramData(df, parameter) :
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

def transformHashingTFData(df, param) :
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

def loadWord2VecModel(path):
    '''
    Load Word2Vec model
        Input : - path
        Output : - model [Word2Vec model data frame]
    '''
    
    model = Word2VecModel.load(path)
    return model

def loadCountVectorizerModel(path):
    '''
    Load Count Vectorizer model
        input : - path
        output: - model [Count Vectorizer model data frame]
    '''
    
    model = CountVectorizerModel.load(path)
    return model

def loadIDFModel(path):
    '''
    Load IDFModel
        input : - path
        output: - model [IDFModel data frame]
    '''
    
    model = IDFModel.load(path)
    return model

def saveModel(model,path):
    '''
    Save model into corresponding path
        Input : - model
                - path
        Output : - saved model
    '''
    
    model.save(path)

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

def getVocab(model) :
    '''
    Get array of terms in the vocabulary.
        Input : - fitted model
        Output : - array of terms
    '''
    return model.vocabulary

# ----------------Testing and Example--------------------#
    
if __name__ == "__main__" :
    
    '''
    Content of elephant.txt :
        An elephant is the biggest living animal on land. 
        It is quite huge in size. 
        It is usually black or grey in colour.
    '''
    
    load_data = loadData("D:/elephant.txt")
    
    #N-Gram
    row = Row(inputTokens=load_data)
    df = spark.createDataFrame([row])
    
    conf = {
            "n" : 2, 
            "inputCol" : "inputTokens", 
            "outputCol" : "nGrams"
            }
    
    transform_df = transformNGramData(df, conf)
    
    #HasingTF
    df_2 = spark.createDataFrame([(load_data,)], ["words"])
    
    conf_2 = {
            "numFeatures" : 100, 
            "inputCol" : "words", 
            "outputCol" : "features"
            }
    
    transform_df2 = transformHashingTFData(df_2, conf_2)
    
    transform_df2.show()
    
    #Count Vectorizer
    df_3 = spark.createDataFrame([
       (0, "a b c".split(" ")),
       (1, "a b b c a".split(" "))
    ], ["id", "words"])
      
    conf_3 = {
            "inputCol" : "words", 
            "outputCol" : "features", 
            "vocabSize" : 3, 
            "minDF" : 2.0
            }
    
    model = countVectorizerTrain(df_3, conf_3)
    
    result = transformData(df_3, model)
    
    result.show(truncate=False)
    
    print(getVocab(model))
    
    #IDF
    df_4 = spark.createDataFrame([(DenseVector([1.0, 2.0]),),
                                (DenseVector([0.0, 1.0]),), 
                                (DenseVector([3.0, 0.2]),)], ["tf"])
    
    conf_4 = {
            "minDocFreq" : 3,
            "inputCol" : "tf",
            "outputCol" : "idf"
            }
    
    model_2 = idfTrain(df_4, conf_4)
    
    transformData(df_4, model_2).show()
    
        
    #Word to Vector
    doc = spark.createDataFrame([(load_data,)], ["sentence"])
    
    conf_5 = {
            "vectorSize" : 5,
            "seed" : 42,
            "inputCol": "sentence",
            "outputCol": "model"
            }
    
    model_3 = word2VecTrain(doc, conf_5)
    model_3.getVectors().show()
    
    transformData(doc, model_3).show()
    
    word = "elephant"
    
    if (foundWord(word, load_data)) :
        getSynonyms(word, load_data, 2, model_3)
    else :
        print(getSynonyms(word, load_data, 2, model_3))
    
    spark.stop()
