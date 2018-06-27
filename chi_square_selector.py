# -*- coding: utf-8 -*-
        
from __future__ import print_function

from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import ChiSqSelector, ChiSqSelectorModel

sc = SparkContext.getOrCreate()
spark = SparkSession(sc)

#define parameter and configuration
parameter = {
        "featuresCol" : "features",
        "labelCol" : "clicked",
        "outputCol" : "selectedFeatures"
        }

num_config = {
        "numTopFeatures" : 50,
        "percentile" : 0.1,
        "fpr" : 0.05
        }

config = {
        "params" : parameter,
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

def transformData(dataFrame, model):
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

#load chiSqSelector model
def loadModel(path):
    '''
        input : - path
        output: - model [ChiSqSelectorModel data frame]
    '''
    model = ChiSqSelectorModel.load(path)
    return model

def copyModel(model):
    '''
    Returning copied model
    Input: model
    Output: model of copy
    '''
    
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
    
    #ChiSquareSelector model
    #trained_model = numChiSqModel(df, "path")
    
    #Save model
    #num_model = saveModel(trained_model, "path")
    
    '''
    - with numTopFeatures selection method
    - numTopFeatures chooses a fixed number of top features 
    according to a chi-squared test.
    '''
    num = numChiSqModel(df, config)
    transformData(df, num).show()
    
    '''
    - with percentile selection method
    - percentile is similar but chooses a fraction of all 
    features instead of a fixed number.
    '''
    #per = perChiSqModel(df, config)
    #transformData(df, per).show()
    
    '''
    - with fpr selection method
    - fpr chooses all features whose p-values are below a threshold, 
    thus controlling the false positive rate of selection.
    '''
    #fpr = fprChiSqModel(df, config)
    #transformData(df, fpr).show()
            
    spark.stop()




    
