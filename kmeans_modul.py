#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 11 01:43:41 2018

@author: Mujirin
email: mujirin@kofera.com
"""
from pyspark.ml.linalg import Vectors
from spark_sklearn.util import createLocalSparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.clustering import KMeansModel
spark = createLocalSparkSession()

def hiperAdapter(hiperparameter):
    '''
    Fungsi ini untuk menyesuaikan config 
    yang tidak lengkap
    ke defaultnya.
    '''
    hiperparameter_default = {'featuresCol':"features", 
          'predictionCol':"prediction", 
          'k':2, 
          'initMode':"k-means||", 
          'initSteps':2, 
          'tol':1e-4, 
          'maxIter':20, 
          'seed':None}
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
    k_means = KMeans(featuresCol = hiperparameter['featuresCol'],
                     predictionCol = hiperparameter['predictionCol'], 
                     k = hiperparameter['k'], 
                     initMode = hiperparameter['initMode'], 
                     initSteps = hiperparameter['initSteps'], 
                     tol = hiperparameter['tol'], 
                     maxIter = hiperparameter['maxIter'], 
                     seed = hiperparameter['seed'])
    model = k_means.fit(df)
    return model

def dfHasil(model):
    '''
    Returning dataframe with its cluster as one of the columns.
    Input:  trained model
    Output: dataframe dengan colom features dan prediction
    '''
    df_hasil = model.summary.predictions
    return df_hasil

def dfCluster(model):
    '''
    Returning the cluster of every data, just one column(prediction).
    Input:  trained model
    Output: dataframe dengan colom prediction
    '''
    df_cluster = model.summary.cluster
    return df_cluster

def dfClusterSize(model):
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

def dfPredict(df, model, featuresCol = 'features', predictionCol = 'prediction'):
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
    model = KMeansModel.load(path)
    return model
    
def dfCostDistance(model, df):
    '''
    Return the K-means cost (sum of squared distances of points 
    to their nearest center) for this model on 
    the given data. 
    (dataset disini bisa diganti dengan dataframe).
    Input:  - model
            - dataframe
    Output: dataframe of cost.
 
    '''
    cost = model.computeCost(df)
    cost = [(Vectors.dense(cost),)]
    df_cost = spark.createDataFrame(cost, ["Cost"])
    return df_cost

def dfClusterCenter(model):
    '''
    Returning cluster center of the model.
    Input:  model
    Output: cluster center dataframe
    '''
    vals1 = model.clusterCenters()
    vals  = []
    for val in vals1:
        vals.append((Vectors.dense(val),))
    df_cluster = spark.createDataFrame(vals, ["Centers"])
    return df_cluster

def kNumber(model):
    k = model.summary.k
    k = [(Vectors.dense(k),)]
    df_k = spark.createDataFrame(k, ["k"])
    return df_k
    
def copyModel(model):
    copy_model = model.copy(extra=None)
    return copy_model

## =============================================================================
## Test and examples
## =============================================================================
#data = [(Vectors.dense([0.0, 0.0]),), 
#        (Vectors.dense([1.0, 1.0]),),
#        (Vectors.dense([9.0, 8.0]),), 
#        (Vectors.dense([8.0, 9.0]),)]
#df = spark.createDataFrame(data, ["features"])
#data2 = [(Vectors.dense([0.0, 0.0]),), 
#        (Vectors.dense([100.0, 100.0]),),
#        (Vectors.dense([900.0, 800.0]),), 
#        (Vectors.dense([800.0, 900.0]),)]
#
#df2 = spark.createDataFrame(data2, ["features"])
#config = {'featuresCol':"features", 
#          'predictionCol':"prediction", 
#          'k':3}
#
#hiperparameter = hiperAdapter(config)
#trained_model = train(df, hiperparameter)
#model = loadModel('kmeans_test')
#c_model = copyModel(trained_model)
#hasil = dfHasil(trained_model)
#print("==========hasil==============")
#hasil.show()
#cluster = dfCluster(trained_model)
#print("===========cluster=============")
#cluster.show()
#cluster_size = dfClusterSize(trained_model)
#print("==========cluster_size==============")
#cluster_size.show()
#predik = dfPredict(df2,model)
#print("==========predik==============")
#predik.show()
#cluster_center = dfClusterCenter(model)
#print("===========cluster_center=============")
#cluster_center.show()
#cost = dfCostDistance(model, df)
#print("===========cost=============")
#cost.show()
#print("===========k=============")
#k = kNumber(trained_model)
#k.show()
