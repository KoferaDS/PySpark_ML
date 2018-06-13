#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 11 01:43:41 2018

@author: Mujirin
email: mujirin@kofera.com
"""
from pyspark.ml.linalg import Vectors
from spark_sklearn.util import createLocalSparkSession
from pyspark.ml.clustering import GaussianMixture
from pyspark.ml.clustering import GaussianMixtureModel
spark = createLocalSparkSession()

def hiperAdapter(hiperparameter):
    '''
    Fungsi ini untuk menyesuaikan config 
    yang tidak lengkap
    ke defaultnya.
    '''
    hiperparameter_default = {"featuresCol":"features", 
                              "predictionCol":"prediction", 
                              "k":2, 
                              "probabilityCol":"probability", 
                              "tol":0.01, 
                              "maxIter":100, 
                              "seed":None}
    hiperparameter_keys = list(hiperparameter.keys())
    hiperparameter_def_keys = list(hiperparameter_default.keys())
    new_hiperparameter = {}
    for hip in hiperparameter_def_keys:
        if hip not in hiperparameter_keys:
            new_hiperparameter[hip] = hiperparameter_default[hip]
        else:
            new_hiperparameter[hip] = hiperparameter[hip]
    return new_hiperparameter


def train(df,hiperparameter):
    '''
    Gaussian Mixture training, returning Gaussian Mixture model.
    input: - Dataframe
           - config (configurasi hiperparameter)
    
    return: kmeans model
    '''
    gm = GaussianMixture(featuresCol = hiperparameter['featuresCol'],
                              predictionCol = hiperparameter['predictionCol'], 
                              k = hiperparameter['k'],
                              probabilityCol = hiperparameter['probabilityCol'], 
                              tol = hiperparameter['tol'], 
                              maxIter = hiperparameter['maxIter'], 
                              seed = hiperparameter['seed'])
    model = gm.fit(df)    
    return model

def summaryCluster(model):
    '''
    Returning dataframe of cluster of every data.
    Input: trained model
    Output: dataframe of prediction(cluster of every data)
    '''
    df_cluster = model.summary.cluster
    return df_cluster

def summaryClusterSize(model):
    '''
    Returning the number or size of each cluster.
    Input:  trained model
    Output: dataframe dengan colom cluster sizes
    '''
    cluster_sizes = model.summary.clusterSizes
    cluster  = []
    for clus in cluster_sizes:
        cluster.append((Vectors.dense(clus),))
    df_cluster_sizes = spark.createDataFrame(cluster, ["Cluster Sizes"])
    return df_cluster_sizes

def summaryResult(model):
    '''
    Returning dataframe with columns feature, prediction, and probability.
    Input: trained model
    Output: dataframe with columns feature, prediction, and probability.
    '''
    df_hasil = model.summary.predictions
    return df_hasil

def summaryLogLikeLihood(model):
    '''
    Returning the log likelihood dataframe
    Input: trained model
    output: log likelihood dataframe
    '''
    log = model.summary.logLikelihood
    log = [(Vectors.dense(log),)]
    df_log = spark.createDataFrame(log, ["Log likelihood"])
    return df_log

def summaryProbability(model):
    '''
    Returning dataframe with columns probability.
    Input: trained model
    Output: dataframe with columns probability.
    '''
    df_probability = model.summary.probability
    return df_probability

def predictDF(df, model, featuresCol = 'features', predictionCol = 'prediction'):
    '''
    Returning dataframe with the coresponding cluster of each data.
    Input:  - model
            - dataframe yang sesuai dengan input training
            - featuresCol: nama colom features, default 'features'
            - predictionCol: nama colom prediction, default 'predicion'
    Output: dataframe dengan colom features dan prediction
    '''
    transformed = model.transform(df).select(featuresCol, predictionCol)
    df_predict = transformed.toDF(featuresCol,predictionCol)
    return df_predict

def saveModel(model,path):
    '''
    Save model into corresponding path.
    Input:  - model
            - path
    Output: saved model
    '''
    model.save(path)
    return

def loadModel(path):
    '''
    Loading model from path.
    Input: path
    Output: loaded model
    '''
    model = GaussianMixtureModel.load(path)
    return model

def copyModel(model):
    '''
    Input: model
    Output: model of copy
    '''
    copy_model = model.copy(extra=None)
    return copy_model

def gaussianModel(model):
    '''
    Retrieve Gaussian distributions as a DataFrame. 
    Each row represents a Gaussian Distribution. 
    The DataFrame has two columns: 
    mean (Vector) and cov (Matrix).
    Input: model
    Output: Gaussian distributions as a DataFrame.
    '''
    df_gauss = model.gaussiansDF
    return df_gauss

def weightModel(model):
    '''
    Returning weight dataframe
    Input: model
    Output: weight dataframe
    '''
    weights_list = model.weights
    weight  = []
    for wei in weights_list:
        weight.append((Vectors.dense(wei),))
    df_weight = spark.createDataFrame(weight, ["Weights"])
    return df_weight
    
## =============================================================================
## Test and examples
## =============================================================================
#data = [(Vectors.dense([-0.1, -0.05 ]),),
#        (Vectors.dense([-0.01, -0.1]),),
#        (Vectors.dense([0.9, 0.8]),),
#        (Vectors.dense([0.75, 0.935]),),
#        (Vectors.dense([-0.83, -0.68]),),
#        (Vectors.dense([-0.91, -0.76]),)]
#df = spark.createDataFrame(data, ["features"])
#
#data2 = [(Vectors.dense([-0.1, -0.05 ]),),
#        (Vectors.dense([-0.01, -0.1]),),
#        (Vectors.dense([0.9, 0.8]),),
#        (Vectors.dense([100.75, 100.935]),),
#        (Vectors.dense([-100.83, -100.68]),),
#        (Vectors.dense([-100.91, -100.76]),)]
#df2 = spark.createDataFrame(data2, ["features"])
#
#config = {'k':3, 
#          'tol':0.0001,
#          'maxIter':10, 
#          'seed':10}
#hiperparameter = hiperAdapter(config)
#
#print("==========config==============")
#print(config)
#print("1==========hiperAdapter==============")
#print(hiperparameter)
#
#print('2======train======')
#trained_model = train(df, hiperparameter)
#print(trained_model.hasSummary)
#
#print('3======save======')
##saveModel(trained_model,'gaussian3')
#
#print('4======load======')
#loaded_model = loadModel('gaussian2')
#print(loaded_model.hasSummary)
#
#print('5======copy======')
#c_model = copyModel(trained_model)
#
#print('6======df_gaussian======')
#df_gaussian = gaussianModel(trained_model)
#df_gaussian.show()
#
#print("7===========summaryCluster=============")
#cluster = summaryCluster(trained_model)
#cluster.show()
#
#print("8==========summaryClusterSize==============")
#cluster_size = summaryClusterSize(trained_model)
#cluster_size.show()
#
#print("11==========summaryLogLikeLihood==============")
#hasil = summaryLogLikeLihood(trained_model)
#hasil.show()
#
#print("13==========summaryHasil==============")
#hasil = summaryResult(trained_model)
#hasil.show()
#
#print("14==========summaryProbability==============")
#dfprob = summaryProbability(trained_model)
#dfprob.show()
#
#print("16==========predik==============")
#predik = predictDF(df2,trained_model)
#predik.show()
#print("==========predik2==============")
#predik = predictDF(df2,loaded_model)
#predik.show()
#print("17==========defWeight==============")
#predik = weightModel(trained_model)
#predik.show()
#print("==========defWeight2==============")
#predik = weightModel(loaded_model)
#predik.show()
