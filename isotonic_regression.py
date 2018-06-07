from __future__ import print_function
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator,ParamGridBuilder, TrainValidationSplit
from pyspark.ml.regression import IsotonicRegression
from pyspark.ml.regression import IsotonicRegressionModel
from pyspark.context import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.sql import Row

sc = SparkContext.getOrCreate()
spark = SparkSession(sc)  

#    Set parameter and its value for Isotonic Regression
isotonic_params = {
                    "predictionCol" : "prediction",
                    "labelCol" : "label",
                    "featuresCol" : "features",
                    "weightCol" : "weight",
                    "isotonic" : True,
                    "featureIndex" : 0
                  }  

#    Set params tuning : - method "crossval" or "trainvalsplit" to train data
#                        - methodParams is set as fold for "crossval" (value : f>0)
#                          and trainratio for "trainvalsplit (value: 0<tr<1)
tune_params = {
                "method" : "trainvalsplit",
                "methodParams" : 5
              }

#    Set configuration whether use tuning/non-tuning
conf1   =  {
                "params" : isotonic_params,
                "tuning" : None
            }

conf2  =   {
                "params" : isotonic_params,
                "tuning" : tune_params
           }

def isotonicRegression(df, conf):
  """ Isotonic Regression training
        Input  : - Dataframe of training (df)
                 - Tuning and hiperparameter configuration (conf)
        output : - Isotonic regression model (model)
  """
  feature_col = conf["params"].get("featuresCol", "features")
  label_col = conf["params"].get("labelCol", "label")
  pred_col = conf["params"].get("predictionCol", "prediction")
  weight_col = conf["params"].get["weightCol", "weight")
  isoton = conf["params"].get("isotonic", True)
  feature_index = conf["params"].get("featureIndex", 0)
      
  ir = IsotonicRegression(featuresCol=feature_col,labelCol=label_col,
                          predictionCol=pred_col, isotonic=isoton, 
                          featureIndex=feature_index)

  if conf["tuning"]:
    if conf["tuning"].get("method").lower() == "crossval":
      folds = conf["tuning"].get("methodParam", 2)
      pg = ParamGridBuilder().build()
      evaluator = RegressionEvaluator(metricName = "r2")
      cv = CrossValidator(estimator=ir, estimatorParamMaps=pg,
                          evaluator=evaluator, numFolds=folds)
      model = cv.fit(df)
    elif conf["tuning"].get("method").lower() == "trainvalsplit":
      tr = conf["tuning"].get("methodParam", 0.8)
      pg = ParamGridBuilder().build()
      evaluator = RegressionEvaluator(metricName = "r2")
      tvs = TrainValidationSplit(estimator=ir, estimatorParamMaps=pg,
                                 evaluator=evaluator, trainRatio=tr)
      model = tvs.fit(df)
  elif conf["tuning"] ==  None:
    model = ir.fit(df)
  return model

def saveModel(model,path):
    '''Save model into corresponding path.
       Input  : - Model
                - Path
       Output : - Saved model
    '''
    model.save(path)
    return
  
def loadzmodel(path):
    '''Loading model from path.
       Input  : - Path
       Output : - Loaded model
    '''
    model = IsotonicRegressionModel.load(path)
    return model

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

def copyModel(model):
    copy_model = model.copy(extra=None)
    return copy_model

# ------------------------------Test and examples--------------------------------

#     Loads dataframe
df_isoton = spark.read.format("libsvm")\
            .load("D:\Kofera\spark-master\data\mllib\sample_isotonic_regression_libsvm_data.txt")    
    
    # Splitting dataframe into dataframe training and test, 
    # ex: 0.6 (60 datainput used as df training and 0.4 used as df test
df_training, df_test = df_isoton.randomSplit([0.6, 0.4], seed=11)
df_training.cache()
   
    # Applied methods to Data
# IR Model
trained_model = isotonicRegression(df_training,conf2)

##Save model
#save = saveModel(trained_model, "path")

##Load model
#load_model = loadModel("path")

# Prediction
testing = predict(df_test,trained_model)
testing.show()

# Root square
r2      = summary_R2(testing, "prediction", "label")  
r2.show() 

# Root mean square
rmse    = summary_Rmse(testing,"prediction", "label")  
rmse.show()
