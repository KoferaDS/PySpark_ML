#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 08:53:53 2018

@author: Mujirin
"""

# =============================================================================
# Principle Componen Analysis
# PCA trains a model to project vectors to a lower dimensional space of the top k principal components.
# =============================================================================
from pyspark.ml.linalg import Vectors
from spark_sklearn.util import createLocalSparkSession
from pyspark.ml.feature import PCA
from pyspark.ml.feature import PCAModel
spark = createLocalSparkSession()

def hiperAdapter(hiperparameter):
    '''
    Fungsi ini untuk menyesuaikan config 
    yang tidak lengkap
    ke defaultnya.
    Input: User hiperparameter setting
    Output: library of hiperparameter with the default
    '''
    hiperparameter_default = {'k':None, 
                              'inputCol':None, 
                              'outputCol':None}
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
    Fits a model to the input dataset with optional parameters.
    Input/Parameters:	
                datafame/dataset – input dataset, which is an instance of pyspark.sql.DataFrame
                config (configurasi hiperparameter)

    Output/Returns:	
                fitted model(s)
    '''
    pca = PCA(k = hiperparameter['k'],
                    inputCol = hiperparameter['inputCol'],
                    outputCol = hiperparameter['outputCol'])
    model = pca.fit(df)
    return model

def saveModel(model,path):
    '''
    Save model into corresponding path.
    Input:  - model
            - path
    Output: saved model
    '''
    model.save(path)
    return

def copyModel(model, extra=None):
    '''
    Creates a copy of this instance with the same uid and some extra params. 
    This implementation first calls Params.copy and 
    then make a copy of the companion Java pipeline 
    component with extra params. 
    So both the Python wrapper and the Java pipeline component get copied.
    Parameters:	model
                extra – Extra parameters to copy to the new instance
    Returns:	Copy of this instance
    '''
    c_model = model.copy(extra)
    return c_model

def loadModel(path):
    '''
    Loading model from path.
    Input: model
    Output: loaded model
    '''
    model = PCAModel.load(path)
    return model

def varianceModel(model):
    '''
    Returns a vector of proportions of 
    variance explained by each 
    principal component.
    Input: Model
    Output: vector dense dataframe of the variance
    '''
    var = model.explainedVariance
    var = [(var,)]
    df_var = spark.createDataFrame(var, ["Variance of the model"])
    return df_var

def paramsExplanation(model):
    '''
    Extracts the embedded default param values and 
    user-supplied values, and then merges them with extra values 
    from input into a flat param map, where the latter value 
    is used if there exist conflicts, i.e., with ordering: default param 
    values < user-supplied values < extra.
    Parameters:	extra – extra param values
    Input: model
    Output: string of explanation of hyperparameter involved
    '''
    param_exp = model.extractParamMap()
    param_exp = [(param_exp,)]
    df_param_exp = spark.createDataFrame(param_exp, ["Parameter explanation of the model"])
    return df_param_exp

def principleComponentsModel(model):
    '''
    Returns a principal components Matrix. Each column is one principal component.
    Input: model
    Output: principal components Matrix dataframe
    '''
    principle = model.pc
    principle = [(principle,)]
    df_param_exp = spark.createDataFrame(principle, ["Principal components matrix"])
    return df_param_exp

def predictDf(model, df, params = None):
    '''
    Transforms the input dataset with optional parameters.
    Returning dataframe with the coresponding cluster of each data.
    Input:  - model
            - dataframe yang sesuai dengan input training
    Output: dataframe dengan colom features dan pca_features.
    '''
    transformed = model.transform(df)
    return transformed


# =============================================================================
# Test and examples
# =============================================================================

print()
print("Data========================")
data = [(Vectors.sparse(5, [(1, 1.0), (3, 7.0)]),),
        (Vectors.dense([2.0, 0.0, 3.0, 4.0, 5.0]),),
        (Vectors.dense([4.0, 0.0, 0.0, 6.0, 7.0]),)]
df = spark.createDataFrame(data,["features"])

data2 = [(Vectors.sparse(5, [(4, 10.0), (3, 7.0)]),),
        (Vectors.dense([20.0, 8.0, 0.3, 400.0, 5.0]),),
        (Vectors.dense([40.0, 10.0, 20.0, 600.0, 700.0]),),
        (Vectors.dense([4.0, 10.0, 200.0, 600.0, 700.0]),),
        (Vectors.dense([3.0, 100.0, 0.0, 6000.0, 7000.0]),)]
df2 = spark.createDataFrame(data2,["features"])

print()
print('data')
print(df.show())
print('data2')
print(df2.show())
print("0. config========================")

config = {'k':2,
          'inputCol':"features",
           'outputCol':"pca_features"
          }
config2 = {'k':4,
          'inputCol':"features",
           'outputCol':"pca_features"
          }
print(config)

print()
print("1. hiperAdapter========================")
hiperparameter = hiperAdapter(config)
hiperparameter2 = hiperAdapter(config2)

print(hiperparameter)
#pca = PCA(k=2, inputCol="features", outputCol="pca_features")
#model = pca.fit(df)

print()
print("2. train========================")
trained_model = train(df, hiperparameter)
trained_model2 = train(df2, hiperparameter2)

#print()
print("3. saveModel========================")
#print(saveModel(trained_model,'pca2'))
#print(saveModel(trained_model2, 'pca3'))

print()
print("4. loadModel========================")
pca1 = loadModel('pca2')
pca2 = loadModel('pca3')

print()
print("5. copyModel========================")
train_cp = copyModel(trained_model)
load_cp = copyModel(pca2)
print(train_cp.transform(df).show())
print(load_cp.transform(df2).show())
#print(df2.show())

print()
print("6. varianceModel========================")
varianceModel_t = varianceModel(trained_model)
varianceModel_lc = varianceModel(pca2)
print(varianceModel_t.show())
print(varianceModel_lc.show())

print()
print("7. paramsExplanation========================")
print(paramsExplanation(train_cp).show())
print(paramsExplanation(load_cp).show())

print()
print("8. principleComponentsModel========================")
print(principleComponentsModel(train_cp).show())
print(principleComponentsModel(load_cp).show())

print()
print("9. predictDf========================")
print(predictDf(train_cp,df).show())
print(predictDf(load_cp,df).show())
print(predictDf(train_cp,df2).show())
print(predictDf(load_cp,df2).show())


print("========finish")

# =============================================================================
# params
# =============================================================================
'''
Param(parent='PCA_4d1387c7b6cac75631f2', name='inputCol', doc='input column name')
Param(parent='PCA_4d1387c7b6cac75631f2', name='k', doc='the number of principal components (> 0)')
Param(parent='PCA_4d1387c7b6cac75631f2', name='outputCol', doc='output column name')
'''