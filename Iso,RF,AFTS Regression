from __future__ import print_function
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import IsotonicRegression,IsotonicRegressionModel
from pyspark.ml.regression import RandomForestRegressor,RandomForestRegressionModel
from pyspark.ml.regression import AFTSurvivalRegression,AFTSurvivalRegressionModel
from pyspark.context import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors


sc = SparkContext.getOrCreate()
spark = SparkSession(sc)  
 
#Set parameter
params = {
                    "predictionCol" : "prediction",
                    "labelCol" : "label",
                    "featuresCol" : "features",
                    "weightCol" : "weight",
                    "isotonic" : True,
                    "featureIndex" : 0
                    
                    "maxDepth" : 5
                    "maxBins" : 32,
                    "minInstancesPerNode" : 1
                    "minInfoGain" : 0.0,
                    "maxMemoryInMB" : 256
                    "cacheNodeIds" : False,
                    "checkpointInterval" : 10
                    "impurity" : "variance",
                    "subsamplingRate" : 1.0
                    "seed" : None
                    "numTrees" : 20
                    "featureSubsetStrategy" : "auto"
                    
                    "censorCol" : "censor",
                    "quantilesCol" : None,
                    "fitIntercept" : True,
                    "maxIter" : 100,
                    "tol" : 1E-6,
                    "quantileProbabilities" : [0.01, 0.05, 0.1, 0.25, 
                                               0.5, 0.75, 0.9, 0.95, 0.99],
                    "aggregationDepth" : 2
                  }           

# Set params tuning
# method : "crossval" , "trainvalsplit"
# methodParams is set as : - fold for "crossval" (value : f>0) 
#                          - trainratio for "trainvalsplit" (value: 0<tr<1)f>0)

tuning_params = {
                    "method" : "trainvalsplit",
                    "methodparams" : 0.8
                }  
# Set configuration whether use tuning/non-tuning
conf1    = {
                "params" : params,
                "tuning" : None
          }
conf2   = {
                "params" : params,
                "tuning" : tuning_params
          }

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
  
def randomforestRegression (df,conf):
    """input  : - Dataframe train (df)
                - Hyperparameter configuration (conf)
       output : - Random Forest Regression Model
    """     
# set params with default value (if value isn't set in params)
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

def saveModel(model,path):
    '''Save model into corresponding path.
       Input  : - Model
                - Path
       Output : - Saved model
    '''
    model.save(path)
    return

def loadModel(path):
    '''Loading model from path.
       Input  : - Path
       Output : - Loaded model
    '''
  # Load model if use crossvalidation tuning
    if conf["tuning"].get("method") == "crossval" :
        loaded_model = CrossValidationModel.load(path)   
  # Load model if use trainvalidationsplit tuning
    elif conf["tuning"].get("method") == "trainval":
        loaded_model = TrainValidationSplitModel.load(path)
  # Load Isotonic Regression model
    elif conf["tuning"].get("method") == None:
        loaded_model = IsotonicRegressionModel.load(path)
  # Load Random Forest Regression model
    elif conf["tuning"].get("method") == None:
        loaded_model = RandomForestRegressionModel.load(path)
  # Load AFT Survival Regression model
    elif conf["tuning"].get("method") == None:
        loaded_model = AFTSurvivalRegressionModel.load(path)
    return loaded_model

def predict(df, model):
    """ Prediction value by the trained model
        Input  : -Dataframe test(df)
                 -Trained model (model)
        Output : -Dataframe with prediction column (transformed)
    """      
    transformed = model.transform(df).select("label","prediction")
    return transformed
    
def prediction(df,model):
    """ show dataframe of prediction in kernel
         Input  : -Dataframe test(df)
                  -Trained model (model)
        Output :  -Dataframe display with prediction column (transformed)
    """
    model.transform(df).show()

def summary_R2(df):
    """ Root square value from the model
        input  : -Dataframe prediction (df)
        output : -Dataframe of R^2 and Rms (df)
    """    
    evaluator = RegressionEvaluator(metricName="r2")
    R2      = evaluator.evaluate(df)
    v_R2    = [(Vectors.dense(R2),)]
    df_R2   = spark.createDataFrame(v_R2, ["R^2"])  
    return df_R2

def summary_Rmse(df):
    """ Root mean square value from the model
        input  : -Dataframe prediction (df)
        output : -Dataframe of R^2 and Rms (df)
    """    
    evaluator = RegressionEvaluator(metricName="rmse")
    Rmse     = evaluator.evaluate(df)
    v_Rmse   = [(Vectors.dense(Rmse),)]
    df_Rmse  = spark.createDataFrame(v_Rmse, ["Rmse"])  
    return df_Rmse

def summary_validationMetrics(model):
    #Only if the model is RandomForestRegressionModel
    """ Validation metrics value
        input : - trained model
        ouput : - validation metrics value
    """
    vm = model.validationMetrics
    return vm
    
def summary_coefficient(model):
    #Only if the model is AFTSurvivalRegressionModel
    """
        input : - trained model
        output : - model coefficient value
    """
    coe = model.coefficients
    return coe

def summary_intercept(model):
    #Only if the model is AFTSurvivalRegressionModel
    """
        input : - trained model
        output : - model intercept value
    """
    intercept = model.intercept
    return intercept

def summary_quantile(model):
    #Only if the model is AFTSurvivalRegressionModel
    """
        input : - trained model
        output : - predicted quantiles value
    """
    pq = model.predictQuantiles
    return pq

def copyModel(model):
    copy_model = model.copy(extra=None)
    return copy_model

# ------------------------------Test and examples--------------------------------

#     Loads dataframe
df_isoton = spark.read.format("libsvm")\
            .load("D:\Kofera\spark-master\data\mllib\sample_isotonic_regression_libsvm_data.txt")    
    
#     Splitting dataframe into dataframe training and test 
#     incase: ratio df_training:df_test = 6:4
df_training, df_test = df_isoton.randomSplit([0.6, 0.4], seed=11)
df_training.cache()
   
#     Applied methods to Data
# IR Model
trained_model = isotonicRegression(df_training,isotonic_params)

#Save model
saved_model = saveModel(trained_model, "path")

#Load model
loaded_model = loadisotonicRegression("path")

# Prediction
testing = predict(df_test,trained_model)
testing.show()

# Root square
r2      = summary_R2(testing)  
r2.show() 
# Root mean square
rmse    = summary_Rmse(testing)  
rmse.show()
