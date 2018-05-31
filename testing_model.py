

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import LinearSVC
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.classification import NaiveBayes
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.classification import MultilayerPerceptronClassifier
from pyspark.ml.classification import OneVsRest



from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.mllib.util import MLUtils

import numpy
from numpy import allclose
from pyspark.sql import Row
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import StringIndexer



sc = SparkContext.getOrCreate()
#sc = SparkContext('local')
spark = SparkSession(sc)



logistic_params = { 
                    "maxIter" : 5, 
                    "regParam" : 0.01, 
                    "elasticNetParam" : 1.0, 
                    "weightCol" : "weight"
                    }


grid = { 
          "numFeatures" : [10, 100, 1000], 
          "regParam" : [0.1, 0.01]
       }


tune_params = { 
                 "method" : "crossval", 
                 "paramGrids" : grid, 
                 "methodParam" : 5  
                }

#without hyperparameter tuning
conf = {   
          "params" : logistic_params,
          "tuning" : None
        }

#example model using hyperparameter tuning
conf2 = {   
          "params" : logistic_params,
          "tuning" : tune_params
        }



def logistic_classifier(df, conf):
  feature_col = conf["params"].get("featuresCol", "features")
  label_col = conf["params"].get("labelCol", "label")
  pred_col = conf["params"].get("predictionCol", "prediction")
  prob_col = conf["params"].get("probabilityCol", "probability")
  
  
  max_iter = conf["params"].get("maxIter", 100)
  reg_param = conf["params"].get("regParam", 0.0)
  elasticNet_param = conf["params"].get("elasticNetParam", 0.0)
  tolr = conf["params"].get("tol", 1e-6)
  fit_intercept = conf["params"].get("fitIntercept", True)
  thres = conf["params"].get("threshold", 0.5)
  thresh = conf["params"].get("thresholds", None)
  std = conf["params"].get("standardization", True)
  weight = conf["params"].get("weightCol", None)
  aggr = conf["params"].get("aggregationDepth", 2)
  fml = conf["params"].get("family", "auto")
  
  
  lr = LogisticRegression(maxIter=max_iter, regParam=reg_param, elasticNetParam=elasticNet_param, \
          tol=tolr, fitIntercept=fit_intercept, threshold=thres, standardization=std, \
            aggregationDepth=aggr, family=fml, weightCol=weight)
    
  if conf["tuning"]:
    if conf["tuning"].get("method").lower() == "crossval":
      logReg = LogisticRegression()
      paramgGrids = conf["tuning"].get("paramGrids")
      folds = conf["tuning"].get("methodParam", 2)
      pg = ParamGridBuilder()
      for key in paramgGrids:
        pg.addGrid(key, paramgGrids[key])
      
      grid = pg.build()
      evaluator = BinaryClassificationEvaluator()
      cv = CrossValidator(estimator=logReg, estimatorParamMaps=grid, evaluator=evaluator, numFolds=folds)
      model = cv.fit(df)
    elif conf["tuning"].get("method").lower() == "trainvalsplit":
      paramgGrids = conf["tuning"].get("paramGrids")
      tr = conf["tuning"].get("methodParam", 0.8)
      pg = ParamGridBuilder()
      for key in paramgGrids:
        pg.addGrid(key, paramgGrids[key])
      
      grid = pg.build()
      evaluator = BinaryClassificationEvaluator()
      tvs = TrainValidationSplit(estimator=lr, estimatorParamMaps=grid, evaluator=evaluator, trainRatio=tr)
      model = tvs.fit(df)

  elif conf["tuning"] == None:
    model = lr.fit(df)
  return model

def predict(df, model):
  result = model.transform(df).head()
  return result.prediction

def prediction(df, model):
  model.transform(df).show()



if __name__ == "__main__":
  
  data = sc.parallelize([
     Row(label=1.0, weight=1.0, features=Vectors.dense(0.0, 5.0)),
     Row(label=0.0, weight=2.0, features=Vectors.dense(1.0, 2.0)),
     Row(label=1.0, weight=3.0, features=Vectors.dense(2.0, 1.0)),
     Row(label=0.0, weight=4.0, features=Vectors.dense(3.0, 3.0))]).toDF()
  
  data2 = spark.createDataFrame([(Vectors.dense([0.0]), 0.0),
      (Vectors.dense([0.4]), 1.0), (Vectors.dense([0.5]), 0.0),
      (Vectors.dense([0.6]), 1.0), (Vectors.dense([1.0]), 1.0)] * 10,
     ["features", "label"])
  
  
  logistic_model = logistic_classifier(data, conf2)
  #print ("model coefficients : ", logistic_model.coefficients)
  #print ("model intercept : ", logistic_model.intercept)
  
  #test0 = sc.parallelize([Row(features=Vectors.dense(-1.0, 0.0, 1.0, 1.0))]).toDF()
  #test1 = sc.parallelize([
   #  Row(label=1.0, weight=1.0, features=Vectors.dense(0.0, 5.0))]).toDF()
  
  #print ("model predict : \n --> ", predict(test1, logistic_model))
  
  
  # sc.stop()
  
  # exec(open("/home/markus/Music/testing_model.py").read())
