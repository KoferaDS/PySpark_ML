# -*- coding: utf-8 -*-
"""
Created on Wed Jun 20 10:52:46 2018

@author: Lenovo
"""

from pyspark.ml.linalg import Vectors
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml import PipelineModel

from pyspark.ml.regression import AFTSurvivalRegression, AFTSurvivalRegressionModel
from pyspark.ml.regression import IsotonicRegression, IsotonicRegressionModel
from pyspark.ml.regression import GBTRegressor, GBTRegressionModel
from pyspark.ml.regression import DecisionTreeRegressor, DecisionTreeModel
from pyspark.ml.regression import LinearRegression, LinearRegressionModel
from pyspark.ml.regression import RandomForestRegressor, RandomForestRegressionModel

from pyspark.ml.feature import VectorIndexer
from pyspark.ml.tuning import (CrossValidator, TrainValidationSplit, ParamGridBuilder)
from pyspark.ml.tuning import (CrossValidatorModel, TrainValidationSplitModel)
from pyspark.ml.evaluation import RegressionEvaluator

# Initialize and create SparkSession instance
sc = SparkContext.getOrCreate()
spark = SparkSession(sc)    
 

#hyperparameter yang di-grid untuk diproses ke ML-tuning (jika ingin membuat model dengan hyperparameter tuning)
grid = { "maxIter" : [50, 100, 120], "regParam" : [0.1, 0.01]}


#--> Setiap parameter yang di-inputkan ke dictionary tidak harus dituliskan, 
#--> jika tidak di-input maka parameter tersebut akan mengambil default-value nya 
#parameter regresi AFT
afts_params = {
                    "predictionCol" : "prediction",
                    "labelCol" : "label",
                    "featuresCol" : "features",
                    "censorCol" : "censor",
                    "quantilesCol" : None,
                    "fitIntercept" : True,
                    "maxIter" : 100,
                    "tol" : 1E-6,
                    "quantileProbabilities" : [0.01, 0.05, 0.1, 0.25, 
                                               0.5, 0.75, 0.9, 0.95, 0.99],
                    "aggregationDepth" : 2
                  }


#parameter regresi linear
linear_params  = { 
                    "maxIter" : 5, 
                    "regParam" : 0.01, 
                    "elasticNetParam" : 1.0, 
                    "tol" : 1e-06, 
                    "fitIntercept" : True, 
                    "standardization" : True, 
                    "solver" : "auto", 
                    "weightCol" : None, 
                    "aggregationDepth" : 2, 
                    "loss" : "squaredError", 
                    "epsilon" : 1.35
                    }


#parameter regresi isotonik
isotonic_params = {
                    "predictionCol" : "prediction",
                    "labelCol" : "label",
                    "featuresCol" : "features",
                    "weightCol" : "weight",
                    "isotonic" : True,
                    "featureIndex" : 0
                  }


#decision tree parameter configuration 
dt_params = {
                     "maxDepth" : 3, 
                     "featuresCol":"features", 
                     "labelCol":"label", 
                     "predictionCol" : "prediction", 
                     "maxBins" : 32, 
                     "minInstancesPerNode" : 1, 
                     "minInfoGain" : 0.0, 
                     "maxMemoryInMB" : 256, 
                     "cacheNodeIds" : False, 
                     "checkpointInterval" : 10, 
                     "impurity" : 'variance', 
                     "seed" : None, 
                     "varianceCol" :None
             }


# Set parameter and its value for randomforest regression
rfr_params = {
                    "featuresCol":"features", 
                    "labelCol":"label", 
                    "predictionCol" : "prediction", 
                    "maxDepth" : 5, 
                    "maxBins" : 32,
                    "minInstancesPerNode" : 1, 
                    "minInfoGain" : 0.0,
                    "maxMemoryInMB" : 256, 
                    "cacheNodeIds" : False,
                    "checkpointInterval" : 10, 
                    "impurity" : "variance",
                    "subsamplingRate" : 1.0, 
                    "seed" : None, 
                    "numTrees" : 20,
                    "featureSubsetStrategy" : "auto"
             }


#parameter regresi gradient boosting tree
gbt_params = {
                    "maxIter" : 20, 
                    "maxDepth" : 3,
                    "featuresCol" : "features", 
                    "labelCol" : "label", 
                    "predictionCol" : "prediction", 
                    "maxBins" : 32,
                    "minInstancesPerNode" : 1, 
                    "minInfoGain" : 0.0, 
                    "maxMemoryInMB" : 256, 
                    "cacheNodeIds" : False, 
                    "subsamplingRate"  : 1.0, 
                    "checkpointInterval" : 10, 
                    "lossType" : "squared", 
                    "stepSize"  : 0.1, 
                    "seed" : None, 
                    "impurity" : "variance"
                }



#tuning parameter -> metode : Cross Validation (crossval) dan Train Validation Split (trainvalsplit)
# "methodParam" untuk crossval : int (bilangan bulat)
# "methodParam" untuk trainval : pecahan antara [0-1]
tune_params = { 
                 "method" : "crossval", 
                 "paramGrids" : grid, 
                 "methodParam" : 3  
                }

#conf1 digunakan apabila model dibuat tanpa hyperparameter-tuning
conf1 = {   
              "params" : rfr_params,
              "tuning" : None
        }


#conf2 digunakan apabila model dibuat dengan parameter tuning, tuning parameter di-inputkan dengan key "tuning" 
conf2 = {   
              "params" : rfr_params,
              "tuning" : tune_params
        }



def converterDF(df, cols, features):
    """
        input : df = dataframe  (per feature per column), 
                cols = variable containing list of feature columns OR 
                       list of feature columns (example : ["a", "b", "c"] )
                features = string (example : "features")
                
        output : dataframe (features in one column)
                 
    """
    converter = VectorAssembler(inputCols=cols,outputCol=features)
    converter_df = converter.transform(df)
    return converter_df


#fungsi regresi AFT
def aftsurvivalRegression(df, conf):
  """ AFT Survival Regression training
        Input  : - Dataframe of training (df)
                 - tuning and hiperparameter configuration (conf)
        output : - AFT survival regression model (model)
  """
  feature_col = conf["params"].get("featuresCol", "features")
  label_col = conf["params"].get("labelCol", "label")
  pred_col = conf["params"].get("predictionCol", "prediction")
  cens_col = conf["params"].get("censorCol", "censor")
  fit_intercept = conf["params"].get("fitIntercept",True)
  max_iter = conf["params"].get("maxIter", 100)
  tol = conf["params"].get("tol", )
  quant_p = conf["params"].get("quantileProbabilities", [0.01, 0.05, 0.1, 0.25, 
                                                        0.5, 0.75, 0.9, 0.95, 0.99])
  quant_col = conf["params"].get("quantilesCol", None)
  agg_depth = conf["params"].get("aggregationDepth", 2)
      
  afts = AFTSurvivalRegression(featuresCol=feature_col,labelCol=label_col,
                          predictionCol=pred_col, censorCol=cens_col,
                          maxIter=max_iter, fitIntercept=fit_intercept,
                          tol=tol, aggregationDepth=agg_depth)

  if conf["tuning"]:
    if conf["tuning"].get("method").lower() == "crossval":
      folds = conf["tuning"].get("methodParam", 2)
      # Set the hiperparameter that we want to grid, incase: maxIter and aggregationDepth
      pg = ParamGridBuilder()\
                .addGrid(afts.maxIter,[10, 50, 100])\
                .addGrid(afts.aggregationDepth[3,4,5])\
                .build()
      evaluator = RegressionEvaluator()
      cv = CrossValidator(estimator=afts, estimatorParamMaps=pg,
                          evaluator=evaluator, numFolds=folds)
      model = cv.fit(df)
      
    elif conf["tuning"].get("method").lower() == "trainvalsplit":
      tr = conf["tuning"].get("methodParam", 0.8)
      # Set the hiperparameter that we want to grid, incase: maxIter and aggregationDepth
      pg = ParamGridBuilder()\
                .addGrid(afts.maxIter,[10, 50, 100])\
                .addGrid(afts.aggregationDepth[3,4,5])\
                .build()
      evaluator = RegressionEvaluator()
      tvs = TrainValidationSplit(estimator=afts, estimatorParamMaps=pg,
                                 evaluator=evaluator, trainRatio=tr)
      model = tvs.fit(df)
  elif conf["tuning"] ==  None:
    model = afts.fit(df)
  return model


#Membuat model menggunakan regresi linear (dari data training)
def linearRegression(df, conf):
    """ 
        input : df [spark.dataframe], conf [configuration params]
        output : linear_regression model [model]
    """
    #memanggil parameter (nilai default)
    featuresCol= conf["params"].get("featuresCol", "features")
    labelCol= conf["params"].get("labelCol", "label")
    predictionCol = conf["params"].get("predictionCol", "prediction")
        
    max_iter = conf["params"].get("maxIter", 100)
    reg_param = conf["params"].get("regParam", 0.0)
    elasticnet_param = conf["params"].get("elasticNetParam", 0.0)
    tol = conf["params"].get("tol", 1e-6)
    fitIntercept = conf["params"].get("fitIntercept", True)
    standardization = conf["params"].get("standardization", True)
    solver = conf["params"].get("solver", "auto")
    weightCol = conf["params"].get("weightCol", None)
    aggregationDepth = conf["params"].get("aggregationDepth", 2)
    loss = conf["params"].get("loss", "squaredError")
    epsilon =  conf["params"].get("epsilon", 1.35)        
        
    lr = LinearRegression(maxIter=max_iter, regParam=reg_param, 
                              elasticNetParam=elasticnet_param)
        
    print ("maxIter : " , lr.getMaxIter())
    print ("regParam : " , lr.getRegParam())
    print ("aggrDepth : " , lr.getAggregationDepth())
        
    #jika menggunakan ml-tuning
    if conf["tuning"]:
            
        #jika menggunakan ml-tuning cross validation  
        if conf["tuning"].get("method").lower() == "crossval":
            paramgGrids = conf["tuning"].get("paramGrids")
            pg = ParamGridBuilder()
            for key in paramgGrids:
              pg.addGrid(key, paramgGrids[key])
          
            grid = pg.build()
            folds = conf["tuning"].get("methodParam")
            evaluator = RegressionEvaluator()
            cv = CrossValidator(estimator=lr, estimatorParamMaps=grid, 
                                evaluator=evaluator, numFolds= folds)
            model = cv.fit(df)
          
        #jika menggunakan ml-tuning train validation split
        elif conf["tuning"].get("method").lower() == "trainvalsplit":
            paramgGrids = conf["tuning"].get("paramGrids")
            pg = ParamGridBuilder()
            for key in paramgGrids:
              pg.addGrid(key, paramgGrids[key])
          
            grid = pg.build()
            tr = conf["tuning"].get("methodParam")
            evaluator = RegressionEvaluator()
            tvs = TrainValidationSplit(estimator=lr, estimatorParamMaps=grid, 
                                       evaluator=evaluator, trainRatio=tr )
            model = tvs.fit(df)
            
    #jika tidak menggunakan ml-tuning
    elif conf["tuning"] == None:
          print ("test")
          model = lr.fit(df)
          
    return model


#fungsi regresi isotonik
def isotonicRegression(df, conf):
  """ Isotonic Regression training
        Input  : - Dataframe of training (df)
        output : - Isotonic regression model (model)
  """
  feature_col = conf["params"].get("featuresCol", "features")
  label_col = conf["params"].get("labelCol", "label")
  pred_col = conf["params"].get("predictionCol", "prediction")
  isoton = conf["params"].get("isotonic",True)
  feature_index = conf["params"].get("featureIndex",0)
      
  ir = IsotonicRegression(featuresCol=feature_col,labelCol=label_col,
                          predictionCol=pred_col, isotonic=isoton, 
                          featureIndex=feature_index)

  model = ir.fit(df)
  return model

  
#Fungsi untuk mendapatkan model dari data (trained model)
def dtRegression(df, conf):
    """ 
        input : df [spark.dataframe], conf [configuration params]
        output : decisiontree_regression model [model]
    """
    featuresCol = conf["params"].get("featuresCol")
    impurity = conf["params"].get("impurity", "variance")
    
    maxDepth    = conf["params"].get("maxDepth", 5)
    maxBin = conf["params"].get("maxBins",32)
    minInstancesPerNode = conf["params"].get("minInstancesPerNode", 1)
    minInfoGain = conf ["params"].get("minInfoGain", 0.0)
    maxMemoryInMB = conf["params"].get("maxMemoryInMB",256)
    cacheNodeIds = conf["params"].get("cacheNodeIds", False)
    checkpointInterval = conf["params"].get("checkpointInterval", 10)
    seed = conf["params"].get("seed", None)
    varianceCol = conf["params"].get("varianceCol", None)   
    
    dt = DecisionTreeRegressor(maxDepth=maxDepth,featuresCol=featuresCol)
    pipeline = Pipeline(stages=[featureIndexer, dt])
    
    print ("maxDepth : " , dt.getMaxDepth())
    
    #jika menggunakan ml-tuning
    if conf["tuning"]:
            
          #jika menggunakan ml-tuning cross validation  
          if conf["tuning"].get("method").lower() == "crossval":
            paramgGrids = conf["tuning"].get("paramGrids")
            pg = ParamGridBuilder()
            for key in paramgGrids:
              pg.addGrid(key, paramgGrids[key])
          
            grid = pg.build()
            folds = conf["tuning"].get("methodParam")
            evaluator = RegressionEvaluator()
            cv = CrossValidator(estimator=pipeline, estimatorParamMaps=grid, 
                                evaluator=evaluator, numFolds= folds)
            model = cv.fit(df)
          
          #jika menggunakan ml-tuning train validation split
          elif conf["tuning"].get("method").lower() == "trainvalsplit":
            paramgGrids = conf["tuning"].get("paramGrids")
            pg = ParamGridBuilder()
            for key in paramgGrids:
              pg.addGrid(key, paramgGrids[key])
          
            grid = pg.build()
            tr = conf["tuning"].get("methodParam")
            evaluator = RegressionEvaluator()
            tvs = TrainValidationSplit(estimator=pipeline, estimatorParamMaps=grid, 
                                       evaluator=evaluator, trainRatio=tr )
            model = tvs.fit(df)
            
    #jika tidak menggunakan ml-tuning
    elif conf["tuning"] == None:
          print ("test")
          model = pipeline.fit(df)
          
    return model


#Fungsi Regresi Random Forest
def randomforestRegression (df,conf):
    """
        input  : - Dataframe train (df)
                - Hyperparameter configuration (conf)
       output : - Random Forest Regression Model
    """     
# set params with default value (if value isn't set in rfr_params)
    feature_col = conf["params"].get("featuresCol", "features")
    label_col = conf["params"].get("labelCol", "label")
    pred_col = conf["params"].get("predictionCol", "prediction")
    max_depth = conf["params"].get("maxDepth", 5)
    num_trees = conf["params"].get("numTrees", 20)
    max_bins= conf["params"].get("maxBins", 32)
    seed = conf["params"].get("seed", None)
    minInstancesPerNode = conf["params"].get("minInstancesPerNode", 1)
    minInfoGain = conf ["params"].get("minInfoGain", 0.0)
    maxMemoryInMB = conf["params"].get("maxMemoryInMB", 256)
    cacheNodeIds = conf["params"].get("cacheNodeIds", False)
    checkpointInterval = conf["params"].get("checkpointInterval", 10)
    impurity = conf["params"].get("impurity", "variance")  
    subSamplingRate = conf["params"].get("subsamplingRate", 1.0)
    featureSubsetStrategy = conf["params"].get("featureSubsetStrategy", "auto")
    
    rfr = RandomForestRegressor(featuresCol=feature_col, labelCol=label_col,
                                predictionCol=pred_col, maxDepth=max_depth,
                                numTrees=num_trees, impurity=impurity)
    
    pipeline = Pipeline(stages=[featureIndexer, rfr])
    if conf["tuning"]:
        if conf["tuning"].get("method").lower() == "crossval":
            folds = conf["tuning"].get("methodParam", 4)
        
# Set the hyperparameter that we want to grid, incase: maxDepth and numTrees
            grid = ParamGridBuilder()\
                .addGrid(rfr.maxDepth,[3,4,5])\
                .addGrid(rfr.numTrees,[15,20])\
                .build()
            evaluator = RegressionEvaluator()
            cv = CrossValidator(estimator=rfr, estimatorParamMaps=grid,
                                evaluator=evaluator, numFolds=folds)
            model = cv.fit(df)
        elif conf["tuning"].get("method").lower() == "trainvalsplit":
            tr = conf["tuning"].get("methodParam", 0.8)
       
# Set the hyperparameter that we want to grid, incase: maxDepth and numTrees
            grid = ParamGridBuilder()\
                .addGrid(rfr.maxDepth,[3,4,5])\
                .addGrid(rfr.numTrees,[15,20])\
                .build()
            evaluator = RegressionEvaluator()
            tvs = TrainValidationSplit(estimator=rfr, estimatorParamMaps=grid,
                                       evaluator=evaluator, trainRatio=tr)
            model = tvs.fit(df)
    elif conf["tuning"] ==  None:
        model = pipeline.fit(df)
    return model


#Fungsi untuk mendapatkan model dari data (trained model)
def gbtRegression(df, conf):
    """ 
        input : df [spark.dataframe], conf [configuration params]
        output : decisiontree_regression model [model]
    """
    featuresCol = conf["params"].get("featuresCol")
    labelCol = conf["params"].get("labelCol")
    predictionCol=conf["params"].get("predictionCol")
    impurity = conf["params"].get("impurity", "variance")
    
    maxDepth    = conf["params"].get("maxDepth", 5)
    maxIter = conf["params"].get("maxIter", 20)
    maxBin = conf["params"].get("maxBins", 32)
    minInstancesPerNode = conf["params"].get("minInstancesPerNode", 1)
    minInfoGain = conf ["params"].get("minInfoGain", 0.0)
    maxMemoryInMB = conf["params"].get("maxMemoryInMB",256)
    cacheNodeIds = conf["params"].get("cacheNodeIds", False)
    subsamplingRate= conf["params"].get("subsamplingRate", 1.0)
    checkpointInterval = conf["params"].get("checkpointInterval", 10)
    lossType = conf["params"].get("lossType", "squared")
    seed = conf["params"].get("seed", None) 
    
    gbt = GBTRegressor(maxIter=maxIter, maxDepth=maxDepth, featuresCol="indexedFeatures")
    pipeline = Pipeline(stages=[featureIndexer, gbt])
    
    print ("maxDepth : " , gbt.getMaxDepth())
    print ("maxIter : ", gbt.getMaxIter())
    
    #jika menggunakan ml-tuning
    if conf["tuning"]:
            
          #jika menggunakan ml-tuning cross validation  
          if conf["tuning"].get("method").lower() == "crossval":
            paramgGrids = conf["tuning"].get("paramGrids")
            pg = ParamGridBuilder()
            for key in paramgGrids:
              pg.addGrid(key, paramgGrids[key])
          
            grid = pg.build()
            folds = conf["tuning"].get("methodParam")
            evaluator = RegressionEvaluator()
            cv = CrossValidator(estimator=pipeline, estimatorParamMaps=grid, 
                                evaluator=evaluator, numFolds= folds)
            model = cv.fit(df)
          
          #jika menggunakan ml-tuning train validation split
          elif conf["tuning"].get("method").lower() == "trainvalsplit":
            paramgGrids = conf["tuning"].get("paramGrids")
            pg = ParamGridBuilder()
            for key in paramgGrids:
              pg.addGrid(key, paramgGrids[key])
          
            grid = pg.build()
            tr = conf["tuning"].get("methodParam")
            evaluator = RegressionEvaluator()
            tvs = TrainValidationSplit(estimator=pipeline, estimatorParamMaps=grid, 
                                       evaluator=evaluator, trainRatio=tr )
            model = tvs.fit(df)
            
    #jika tidak menggunakan ml-tuning
    elif conf["tuning"] == None:
          print ("test")
          model = pipeline.fit(df)
          
    return model


#Menampilkan tree model dari model decision tree (ket : dapat dipanggil apabila tidak menggunakan ML-Tuning)
def dtModel(model):
    """
        input : model (DecisionTreeModel/PipelineModel)
        output : the depth and nodes of DecisionTreeRegressionModel
    """
    
    tModel = model.stages[1]
    return tModel


#Menampilkan tree model dari model GBT (ket : dapat dipanggil apabila tidak menggunakan ML-Tuning)
def gbtModel(model):
    """
        input    : model 
        output  : depth and nodes of tree
    """
    gbModel = model.stages[1]
    return gbModel     


#Menampilkan validator metri (jika menggunakan ML-Tuning)
def validatorMetrics(model):
    """
        input : model (TrainValidationSplitModel)
        output : validation metrics 
    """
    
    vm = model.validationMetrics
    return vm


#Menampilkan average metrics dari CrossValidator Model
def avgMetrics(model):
    """
        input    : CrossValidatorModel
        output  : metrics
    """
    avm = model.avgMetrics
    return avm 


#menyimpan model
def saveModel(model, path):
    """
        input : model(CrossValidatorModel / TrainValidationSplitModel / LinearRegressionModel)
        output : None
    """
    model.save(path)
    

#Memuat model
def loadaftRegression(conf, path):
    '''Loading model from path.
       Input  : - Path
       Output : - Loaded model
    '''
  # Load model if use crossvalidation tuning
    if conf["tuning"].get("method") == "crossval" :
        loaded_model = CrossValidatorModel.load(path)   
  # Load model if use trainvalidationsplit tuning
    elif conf["tuning"].get("method") == "trainval":
        loaded_model = TrainValidationSplitModel.load(path)
  # Load model if non-tuning
    elif conf["tuning"].get("method") == None:
        loaded_model = AFTSurvivalRegressionModel.load(path)
    return loaded_model


#Load Model Linear Regression (jika tidak menggunakan ML-tuning = conf1, jika menggunakan ML-tuning = conf2 )
def loadlinearRegression(conf, path):
    """
        input : conf, path
        output : model
                (CrossValidatorModel / TrainValidationSplitModel / LinearRegressionModel)
    """
                   
    #Jika menggunakan ML-Tuning
    if conf["tuning"]:    
        #Jika menggunakan Cross Validation, maka tipe model = CrossValidatorModel
        if conf["tuning"].get("method").lower() == "crossval":
            loaded_model = CrossValidatorModel.load(path)        
        #Jika menggunakan Train Validation, maka tipe model = TrainValidationSplitModel   
        elif conf["tuning"].get("method").lower() == "trainvalsplit":
            loaded_model = TrainValidationSplitModel.load(path)
    
    #Jika tidak menggunakan ML-tuning, tipe model = LinearRegressionModel    
    elif conf["tuning"] == None:
        loaded_model = LinearRegressionModel.load(path)
    
    return loaded_model


def loadisotonicRegression(path):
    '''Loading model from path.
       Input  : - Path
       Output : - Loaded model
    '''
    loaded_model = IsotonicRegressionModel.load(path)
    return loaded_model


#Load Model Decision Tree Regression (jika tidak menggunakan ML-tuning = conf1, jika menggunakan ML-tuning = conf2 )
def loaddtRegression(conf, path):
    """
        input : conf, path
        output : model (CrossValidatorModel / TrainValidationSplitModel / PipelineModel)
    """
    
    #Jika menggunakan ML-Tuning
    if conf["tuning"]:    
        #Jika menggunakan Cross Validation, maka tipe model = CrossValidatorModel
        if conf["tuning"].get("method").lower() == "crossval":
            loaded_model = CrossValidatorModel.load(path)        
        #Jika menggunakan Train Validation, maka tipe model = TrainValidationSplitModel   
        elif conf["tuning"].get("method").lower() == "trainvalsplit":
            loaded_model = TrainValidationSplitModel.load(path)
    
    #Jika tidak menggunakan ML-tuning, tipe model = PipelineModel    
    elif conf["tuning"] == None:
        loaded_model = PipelineModel.load(path)
    
    return loaded_model

   
def loadrfRegression(conf, path):
    '''Loading model from path.
       Input  : -Path
       Output : -Loaded model
    '''
  # Loaded model if use crossvalidation tuning
    if conf["tuning"].get("method") == "crossval" :
        loaded_model = CrossValidatorModel.load(path)   
  # Loaded model if use trainvalidationsplit tuning
    elif conf["tuning"].get("method") == "trainval":
        loaded_model = TrainValidationSplitModel.load(path)
  # Loaded model if use pipeline (non-tuning)
    elif conf["tuning"].get("method") == None:
        loaded_model = PipelineModel.load(path)
        
    return loaded_model


#Load Model GBT Regression (jika tidak menggunakan ML-tuning = conf1, jika menggunakan ML-tuning = conf2 )
def loadgbtRegression(conf, path):
    """
        input : conf, path
        output : model (CrossValidatorModel / TrainValidationSplitModel / PipelineModel)
    """    
    #Jika menggunakan ML-Tuning
    if conf["tuning"]:    
        #Jika menggunakan Cross Validation, maka tipe model = CrossValidatorModel
        if conf["tuning"].get("method").lower() == "crossval":
            loaded_model = CrossValidatorModel.load(path)        
        #Jika menggunakan Train Validation, maka tipe model = TrainValidationSplitModel   
        elif conf["tuning"].get("method").lower() == "trainvalsplit":
            loaded_model = TrainValidationSplitModel.load(path)
    
    #Jika tidak menggunakan ML-tuning, tipe model = PipelineModel    
    elif conf["tuning"] == None:
        loaded_model = PipelineModel.load(path)
    
    return loaded_model

#menampilkan prediksi test data, jika menggunakan model regresi linear
def predict(df, model):
    """ 
        input   : df [spark.dataframe], linear_regression model [model]
        output  : prediction [dataframe]
    """    
    val = model.transform(df)
    prediction = val.select("label", "prediction")
    return prediction   

#menunjukkan nilai R-square 
def summaryR2(df, predictionCol="prediction", labelCol="label"):
    """ 
        input : df [spark.dataframe]
        output : R squared on test data [float]
    """    
    lr_evaluator = RegressionEvaluator(predictionCol=predictionCol, 
             labelCol=labelCol, metricName="r2")
    r2 =  lr_evaluator.evaluate(df)
    r2 = [(Vectors.dense(r2),)]
    r2_df = spark.createDataFrame(r2, ["R-square"])
    return r2_df


#menunjukkan nilai rms
def summaryRMSE(df, predictionCol="prediction", labelCol="label"):
    """ 
        input : df [spark.dataframe]
        output : RMS on test data [float]
    """    
    lr_evaluator = RegressionEvaluator(predictionCol=predictionCol, 
             labelCol=labelCol, metricName="rmse")
    rmse =  lr_evaluator.evaluate(df)
    rmse = [(Vectors.dense(rmse),)]
    rmse_df = spark.createDataFrame(rmse, ["RMS"])
    return rmse_df 
    
#memilih hasil pada baris tertentu (prediksi)
def rowSlicing(df, n):
  """ dataFrame to slicedDataFrame"""
  num_of_data = df.count()
  ls = df.take(num_of_data)
  df = sc.parallelize([ls[n]]).toDF()
  return df

#menyalin model
def copyModel(model):
    copy_model = model.copy(extra=None)
    return copy_model



#>>>======================================(example of usage)==============================================<<<


#Load the input data
#Loads dataframe untuk isotonic regression
isoton_df = spark.read.format("libsvm")\
            .load("C:/Users/Lenovo/spark-master/data/mllib/sample_isotonic_regression_libsvm_data.txt")   


#load data untuk linear regression
#linear_df = spark.read.format("libsvm")\
#        .load("C:/Users/Lenovo/spark-master/data/mllib/sample_linear_regression_data.txt")

                                 
                
#load data untuk Decision Tree, Random forest, dan GBT
#Dataframe input
tree_df = spark.read.format("libsvm").load("C:/Users/Lenovo/spark-master/data/mllib/sample_libsvm_data.txt")


#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
#FEATURE INDEXER HANYA DIAKTIFKAN BILA MENGGUNAKAN "DECISION TREE", "RANDOM FOREST", DAN "GradientBoostingTree"
#!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!


# Automatically identify categorical features, and index them.
# We specify maxCategories so features with > 4 distinct values are treated as continuous.
featureIndexer =\
        VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(tree_df)

training, test = tree_df.randomSplit([0.7, 0.3], seed = 11)


#CONTOH PENGGUNAAN FUNGSI

#create model
model = randomforestRegression(training, conf1)

#use the model for prediction
testing = predict(test, model)

#accuracy metric : R-square
rsq = summaryR2(testing, "prediction", "label")  

#accuracy metric : RMS
rms= summaryRMSE(testing, "prediction", "label")
