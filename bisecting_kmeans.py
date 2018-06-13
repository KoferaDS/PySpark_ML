#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 11 01:43:41 2018

@author: Mujirin
email: mujirin@kofera.com
"""
# =============================================================================
# Bisecting KMeans
# =============================================================================
'''
A bisecting k-means algorithm based on the paper “A comparison of document 
clustering techniques” by Steinbach, Karypis, and Kumar, 
with modification to fit Spark. The algorithm starts from 
a single cluster that contains all points. Iteratively it finds 
divisible clusters on the bottom level and bisects each of them 
using k-means, until there are k leaf clusters in total or no 
leaf clusters are divisible. The bisecting steps of clusters on 
the same level are grouped together to increase parallelism. 
If bisecting all divisible clusters on the bottom level would 
result more than k leaf clusters, larger clusters get higher 
priority.
'''
from pyspark.ml.linalg import Vectors
from pyspark.ml.clustering import BisectingKMeans
from pyspark.ml.clustering import BisectingKMeansModel
from spark_sklearn.util import createLocalSparkSession
spark = createLocalSparkSession()

def hiperAdapter(hiperparameter):
    '''
    Fungsi ini untuk menyesuaikan config 
    yang tidak lengkap
    ke defaultnya.
    Input: User hiperparameter setting
    Output: library of hiperparameter with the default
    '''
    hiperparameter_default = {'featuresCol':"features", 
                              'predictionCol':"prediction", 
                              'maxIter':20, 
                              'seed':None, 
                              'k':4, 
                              'minDivisibleClusterSize':1.0}
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
    KMeans training, returning KMeans model.
    input: - Dataframe
           - config (configurasi hiperparameter)
    
    return: kmeans model
    '''
    bs_kmeans = BisectingKMeans(featuresCol = hiperparameter['featuresCol'],
                          predictionCol = hiperparameter['predictionCol'],
                          maxIter = hiperparameter['maxIter'],
                          seed = hiperparameter['seed'],
                          k = hiperparameter['k'],
                          minDivisibleClusterSize = hiperparameter['minDivisibleClusterSize']
                          )
    model = bs_kmeans.fit(df)
    return model
def summaryCluster(model):
    '''
    Returning the cluster of every data, just one column(prediction).
    Input:  trained model
    Output: dataframe dengan colom prediction
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
    Returning dataframe with its cluster as one of the columns.
    Input:  trained model
    Output: dataframe dengan colom features dan prediction
    '''
    df_hasil = model.summary.predictions
    return df_hasil

def predictDF(model, df, featuresCol = 'features', predictionCol = 'prediction'):
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

def copyModel(model):
    '''
    Creates a copy of this instance with the same uid 
    and some extra params. This implementation first calls 
    Params.copy and then make a copy of the companion Java pipeline component with extra params. So both the Python wrapper 
    and the Java pipeline component get copied.
    '''
    copy_model = model.copy(extra=None)
    return copy_model

def loadModel(path):
    '''
    Loading model from path.
    Input: model
    Output: loaded model
    ***catatan:
        ini kasus yang sangat berbeda, load model membutuhkan model
        mungkin karena yang buat di pyspark lupa kalau load harusnya di LDAModel tempatnya, atau ada faktor lain
    '''
    model = BisectingKMeansModel.load(path)
    return model


def clusterCenterModel(model):
    '''
    Get the cluster centers, represented as a list of NumPy arrays.
    but in this module its transform to dataframe
    Input:  model
    Output: cluster center dataframe
    '''
    vals1 = model.clusterCenters()
    vals  = []
    for val in vals1:
        vals.append((Vectors.dense(val),))
    df_cluster = spark.createDataFrame(vals, ["Cluster Centers"])
    return df_cluster

def costDistanceDf(model, df):
    '''
    Computes the sum of squared distances 
    between the input points and their 
    corresponding cluster centers.
    Input:  - model
            - dataframe
    Output: dataframe of cost.
    '''
    cost = model.computeCost(df)
    cost = [(Vectors.dense(cost),)]
    df_cost = spark.createDataFrame(cost, ["Square distance"])
    return df_cost

def paramsExplanation(model):
    '''
    Input: model
    Output: string of explanation of hyperparameter involved
    '''
    param_exp = model.extractParamMap()
    param_exp = [(param_exp,)]
    df_param_exp = spark.createDataFrame(param_exp, ["Parameter explanation of the model"])
    return df_param_exp

## =============================================================================
## Test and examples
## =============================================================================
#
#print()
#print("Data========================")
#data = [(Vectors.dense([0.0, 0.0]),), 
#        (Vectors.dense([1.0, 1.0]),),
#        (Vectors.dense([9.0, 8.0]),),
#        (Vectors.dense([8.0, 9.0]),)]
#df = spark.createDataFrame(data, ["features"])
#data2 = [(Vectors.dense([1000.0, 10000.0]),), 
#        (Vectors.dense([1.0, 1000.0]),),
#        (Vectors.dense([9000.0, 8000.0]),),
#        (Vectors.dense([8.0, 9000.0]),)]
#df2 = spark.createDataFrame(data2, ["features"])
#print()
#print('data')
#print(df.show())
#print('data2')
#print(df2.show())
#print("0. config========================")
#
#config = {'k':2,
#          'minDivisibleClusterSize':1.0
#          }
#
#print(config)
#
#print()
#print("1. hiperAdapter========================")
#hiperparameter = hiperAdapter(config)
#print(hiperparameter)
#
#print()
#print("2. train========================")
#trained_model = train(df, hiperparameter)
#trained_model2 = train(df2, hiperparameter)
#
#print()
#print("3. clusterCenterModel========================")
#print(clusterCenterModel(trained_model).show())
#print(clusterCenterModel(trained_model2).show())
#
#print()
#print("4. costDistanceDf========================")
#print(costDistanceDf(trained_model,df).show())
#print(costDistanceDf(trained_model2,df2).show())
#
#print()
#print("5. copyModel========================")
#print(copyModel(trained_model))
#print(copyModel(trained_model2))
#
#print()
#print("6. paramsExplanation========================")
#print(paramsExplanation(trained_model).show())
#print(paramsExplanation(trained_model2).show())
#
#print()
#print("7. saveModel========================")
##print(saveModel(trained_model,'bs1'))
##print(saveModel(trained_model2, 'bs2'))
#
#print()
#print("8. loadModel========================")
#bs1 = loadModel('bs1')
#bs2 = loadModel('bs2')
##print(clusterCenterModel(bs1).show())
##print(clusterCenterModel(bs2).show())
#print()
#print("9. summaryCluster========================")
#print(summaryCluster(trained_model).show())
#print(summaryCluster(trained_model2).show())
#
#print()
#print("10. paramsExplanation========================")
#print(paramsExplanation(trained_model).show())
#print(paramsExplanation(trained_model2).show())
#
#print()
#print("14. summaryHasil========================")
#print(summaryResult(trained_model).show())
#print(summaryResult(trained_model2).show())
#
#print()
#print("14. predictDf========================")
#print(predictDF(trained_model,df2).show())
#print(predictDF(trained_model2,df2).show())
#
#
#print("========finish")
#
## =============================================================================
## params
## =============================================================================
#'''
#Param(parent='BisectingKMeans_476f91005414f2094c7c', 
#    name='k', 
#    doc='The desired number of leaf clusters. 
#    Must be > 1.')
#Param(parent='BisectingKMeans_476f91005414f2094c7c', 
#    name='featuresCol', 
#    doc='features column name')
#Param(parent='BisectingKMeans_48708b5db69ae03cb9bb', 
#    name='maxIter', 
#    doc='maximum number of iterations (>= 0)')
#Param(parent='BisectingKMeans_48708b5db69ae03cb9bb', 
#    name='minDivisibleClusterSize', 
#    doc='The minimum number of points (if >= 1.0) 
#    or the minimum proportion of points 
#    (if < 1.0) of a divisible cluster.')
#Param(parent='BisectingKMeans_48708b5db69ae03cb9bb', 
#    name='predictionCol', 
#    doc='prediction column name')
#Param(parent='BisectingKMeans_48708b5db69ae03cb9bb', 
#    name='seed', 
#    doc='random seed')
#
#'''