from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator,ParamGridBuilder, TrainValidationSplit
from pyspark.ml.tuning import CrossValidatorModel, TrainValidationSplitModel
from pyspark.ml.regression import AFTSurvivalRegression, AFTSurvivalRegressionModel
from pyspark.context import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

sc = SparkContext.getOrCreate()
spark = SparkSession(sc)  
 
# Set parameter and its value for AFT Survival Regression
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
    
grid = {
        "maxIter" : [10,50,100],
        "aggregationDepth" : [3,4,5]
        }        
# Set params tuning
# method : "crossval" , "trainvalsplit"
# methodParams is set as : - fold for "crossval" (value : f>0) 
#                          - trainratio for "trainvalsplit" (value: 0<tr<1)
tune_params = {
                "method" : "trainvalsplit",
                "paramGrids" : grid,
                "methodParams" : 0.6
              }

# Set configuration whether use tuning/non-tuning
conf   =  {
                "params" : afts_params,
                "tuning" : None
            }

conf2  =   {
                "params" : afts_params,
                "tuning" : tune_params
           }

def convertDF(df,col,featuresCol):
    """ Convert with assemble multiple features column into single column
        Input : - Dataframe input (df)
                - List containing of features column (list)
                - Features output column name (string) 
        Output: - Dataframe of assembled features column (df)
    """
    convert = VectorAssembler(inputCol=col,
                              outputCol=featuresCol)
    final_convert = convert.transform(df)
    return final_convert

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
      paramGrids = conf["tuning"].get("paramGrids")
      pg=ParamGridBuilder()
      for key in paramGrids:
          pg.addGrid(key, paramGrids[key])
      grid = pg.build()
      evaluator = RegressionEvaluator()
      cv = CrossValidator(estimator=afts, estimatorParamMaps=grid,
                          evaluator=evaluator, numFolds=folds)
      model = cv.fit(df)
      
    elif conf["tuning"].get("method").lower() == "trainvalsplit":
      tr = conf["tuning"].get("methodParam", 0.8)
      # Set the hiperparameter that we want to grid, incase: maxIter and aggregationDepth
      paramGrids = conf["tuning"].get("paramGrids")
      pg=ParamGridBuilder()
      for key in paramGrids:
          pg.addGrid(key, paramGrids[key])
      grid = pg.build()
      evaluator = RegressionEvaluator()
      tvs = TrainValidationSplit(estimator=afts, estimatorParamMaps=grid,
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
        loaded_model = CrossValidatorModel.load(path)   
  # Load model if use trainvalidationsplit tuning
    elif conf["tuning"].get("method") == "trainval":
        loaded_model = TrainValidationSplitModel.load(path)
  # Load model if non-tuning
    elif conf["tuning"].get("method") == None:
        loaded_model = AFTSurvivalRegressionModel.load(path)
    return loaded_model

def predict(df, model):
    """ Prediction value by the trained model
        Input  : -Dataframe test(df)
                 -Trained model (model)
        Output : -Dataframe with prediction column (transformed)
    """      
    transformed = model.transform(df).select("label","prediction","censor")
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
  
df_afts = spark.read.format("libsvm")\
            .load("C:/Users/Lenovo/spark-master/data/mllib/sample_libsvm_data.txt")    

#    Assemble dataframe features into one column
#    Set list of column name from the dataframe
column = []
#    Set assembled column name
features_name= ["features"]
#   Set converted dataframe
build = convertDF(df_afts,column,features_name)    
              
#     Splitting dataframe into dataframe training and test 
#     incase: 0.6 (70%) datainput used as df training and 0.4 (30%) used as df test
df_training, df_test = build.randomSplit[0.7, 0.3]
df_training.cache()
   
#     Applied methods to Data
# AFT Survival Model
trained_model = aftsurvivalRegression(df_training,conf2)

##Save model
#save = saveModel(trained_model, "path")
#
##Load model
#loaded_model = loadModel("path")
#
##Prediction
#testing = predict(df_test,trained_model)
#testing.show()
#
##Root square
#r2      = summary_R2(testing)  
#r2.show() 
##Root mean square
#rmse    = summary_Rmse(testing)  
#rmse.show()
##Model Coefficient
#coeff = summary_coefficient(trained_model)
#coeff.show()
##Model Intercept
#interc = summary_intercept(trained_model)
#interc.show()
##QuantilePrediction
#qp = summary_quantile(trained_model)
#qp.show()
