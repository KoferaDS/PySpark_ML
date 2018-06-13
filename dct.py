#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 31 08:53:53 2018

@author: Mujirin
"""

# =============================================================================
# Discrete Cosine Transform (DCT)
# A feature transformer that takes the 1D discrete cosine transform of a real vector. No zero padding is performed on the input vector. It returns a real vector of the same length representing the DCT. The return vector is scaled such that the transform matrix is unitary (aka scaled DCT-II).
# =============================================================================
from pyspark.ml.linalg import Vectors
from spark_sklearn.util import createLocalSparkSession
from pyspark.ml.feature import DCT
spark = createLocalSparkSession()

def hiperAdapter(hiperparameter):
    '''
    Fungsi ini untuk menyesuaikan config 
    yang tidak lengkap
    ke defaultnya.
    Input: User hiperparameter setting
    Output: library of hiperparameter with the default
    '''
    hiperparameter_default = {'inverse':False, 
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

def DCTTransform(df,hiperparameter):
    '''
    Transforms the input dataset with optional parameters.
    Parameters:	
        dataset – input dataset, which is an instance of pyspark.sql.DataFrame
        params – an optional param map that overrides embedded params.
    Returns:	
        transformed dataset with coloumn is inputCol and the output column is outputCol
    '''
    dct = DCT(inverse = hiperparameter['inverse'],
              inputCol = hiperparameter['inputCol'],
              outputCol = hiperparameter['outputCol'])
    df_transformed = dct.transform(df)
    return df_transformed

def saveDF(df,path):
    '''
    Save model into corresponding path.
    Input:  - model
            - path
    Output: saved model
    '''
    df.write.save(path)
    return



## =============================================================================
## Test and examples
## =============================================================================
#
#print()
#print("Data========================")
#df1 = spark.createDataFrame([(Vectors.dense([5.0, 8.0, 6.0]),)], ["vec"])
#df12 = spark.createDataFrame([(Vectors.dense([5.0, 8.0, 6.0, 600,700,800,900,2090]),)], ["vec"])
#
#print()
#print('data')
#print(df1.show())
#print('data2')
#print(df12.show())
#print("0. config========================")
#
#config = {'inverse':False,
#          'inputCol':"vec", 
#          'outputCol':"resultVec"}
#
#print(config)
#
#print()
#print("1. hiperAdapter========================")
#hiperparameter = hiperAdapter(config)
#
#print(hiperparameter)
#
#
#print()
#print("2. DCTTransform========================")
#df2 = DCTTransform(df1, hiperparameter)
#df22 = DCTTransform(df12, hiperparameter)
#print(df2.show())
#print(df22.show())
##print()
#print("3. savedataframe========================")
##print(saveDF(df2,'lda2'))
##print(saveDF(df22, 'lda3'))
#
#
##print("========finish")
##
### =============================================================================
### params
### =============================================================================
##'''
##outputCol = Param(parent='undefined', name='outputCol', doc='output column name.')
##inputCol = Param(parent='undefined', name='inputCol', doc='input column name.')
##inverse = Param(parent='undefined', name='inverse', doc='Set transformer to perform inverse DCT, default False.')
##
##'''