# -*- coding: utf-8 -*-
        
from __future__ import print_function

from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import ChiSqSelector, ChiSqSelectorModel

sc = SparkContext.getOrCreate()
spark = SparkSession(sc)

#define parameter and configuration
param = {
        "featuresCol" : "features",
        "labelCol" : "clicked",
        "outputCol" : "selectedFeatures"
        }

num_config = {
        "numTopFeatures" : 3,
        "percentile" : 0.5,
        "fpr" : 0.5
        }

config = {
        "params" : param,
        "selectedType" : num_config
        }
    
#create chiSquareSelector with numTopFeatures selector method    
def numChiSqModel(df, conf):
    '''
        - input: - df [spark.dataFrame]
                 - conf [configuration params]
        - output: - generalized linear regression model [model]
    '''
    label_col = conf["params"].get("labelCol")
    output_col = conf["params"].get("outputCol")
    features_col = conf["params"].get("featuresCol")
      
    selector = ChiSqSelector(featuresCol=features_col,
                             outputCol=output_col, labelCol=label_col)
    
    selector.setSelectorType("numTopFeatures")
    selector.setNumTopFeatures(conf["selectedType"].get("numTopFeatures"))
    
    model = selector.fit(df)
    
    #print("Selector Type : %s " % selector.getSelectorType())
    #print("ChiSqSelector output with top %d features selected" % selector.getNumTopFeatures())
    
    return model

#create chiSquareSelector with percentile selector method
def perChiSqModel(df, conf):
    '''
        - input: - df [spark.dataFrame]
                 - conf [configuration params]
        - output: - generalized linear regression model [model]
    '''
    label_col = conf["params"].get("labelCol")
    output_col = conf["params"].get("outputCol")
    features_col = conf["params"].get("featuresCol")
      
    selector = ChiSqSelector(featuresCol=features_col,
                             outputCol=output_col, labelCol=label_col)
    
    selector.setSelectorType("percentile")
    selector.setPercentile(conf["selectedType"].get("percentile"))
    
    model = selector.fit(df)
    
    #print("Selector Type : %d" % selector.getPercentile())
    #print("ChiSqSelector output with top %s features selected" % selector.getSelectorType())
    
    return model

#create chiSquareSelector with fpr selector method
def fprChiSqModel(df, conf):
    '''
        - input: - df [spark.dataFrame]
                 - conf [configuration params]
        - output: - generalized linear regression model [model]
    '''
    label_col = conf["params"].get("labelCol")
    output_col = conf["params"].get("outputCol")
    features_col = conf["params"].get("featuresCol")
      
    selector = ChiSqSelector(featuresCol=features_col,
                             outputCol=output_col, labelCol=label_col)
    
    selector.setSelectorType("fpr")
    selector.setFpr(conf["selectedType"].get("fpr"))
    
    model = selector.fit(df)
    
    #print("Selector Type : %d" % selector.getFpr())
    #print("ChiSqSelector output with top %s features selected" % selector.getSelectorType())
    
    return model

def numTransformModel(dataFrame, conf):
    model = numChiSqModel(dataFrame, conf)
    return model.transform(dataFrame)

def perTransformModel(dataFrame, conf):
    model = perChiSqModel(dataFrame, conf)
    return model.transform(dataFrame)

def fprTransformModel(dataFrame, conf):
    model = fprChiSqModel(dataFrame, conf)
    return model.transform(dataFrame)

#Save model into corresponding path    
def saveModel(model,path):
    '''
    Save model into corresponding path
    Input : - model
            - path
    Output : -saved model
        
    '''
    model.save(path)

#load chiSqSelector selector
def loadModel(path):
    '''
        Loading model from path
        input : - path  
        output : - loaded model
    '''
    selector = ChiSqSelector.load(path)    
    return selector

#load chiSqSelector model
def loadData(path):
    '''
        input : - path
        output: - model [ChiSqSelectorModel data frame]
    '''
    model = ChiSqSelectorModel.load(path)
    return model

def copyModel(model):
    copy_model = model.copy(extra=None)
    return copy_model

# ----------------Testing and Example--------------------#

if __name__ == "__main__" :
    
    #Creating a dataframe
    df = spark.createDataFrame([
            (7, Vectors.dense([0.0, 0.0, 18.0, 1.0]), 1.0,),
            (8, Vectors.dense([0.0, 1.0, 12.0, 0.0]), 0.0,),
            (9, Vectors.dense([1.0, 0.0, 15.0, 0.1]), 0.0,)],
            ["id", "features", "clicked"])
    
    '''
    - with numTopFeatures selection method
    - numTopFeatures chooses a fixed number of top features 
    according to a chi-squared test.
    '''
    #ChiSquareSelector model
    #trained_model = numChiSqModel(df, "path")
    
    #Save model
    #num_model = saveModel(trained_model, "path")
          
    print(numChiSqModel(df, config))
    num_model = numTransformModel(df, config)
    print(num_model)
    num_model.show()
    
    
    '''
    - with percentile selection method
    - percentile is similar but chooses a fraction of all 
    features instead of a fixed number.
    '''
    #per_model = perTransformModel(df, config)
    #per_model.show()
    
    '''
    - with fpr selection method
    - fpr chooses all features whose p-values are below a threshold, 
    thus controlling the false positive rate of selection.
    '''
    #fpr_model = fprTransformModel(df, config)
    #fpr_model.show()
            
    spark.stop()




    
