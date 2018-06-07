from __future__ import print_function
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.regression import IsotonicRegression
from pyspark.ml.regression import IsotonicRegressionModel
from pyspark.context import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors


sc = SparkContext.getOrCreate()
spark = SparkSession(sc)  
 
#Set isotonic parameter
isotonic_params = {
                    "predictionCol" : "prediction",
                    "labelCol" : "label",
                    "featuresCol" : "features",
                    "weightCol" : "weight",
                    "isotonic" : True,
                    "featureIndex" : 0
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

def saveModel(model,path):
    '''Save model into corresponding path.
       Input  : - Model
                - Path
       Output : - Saved model
    '''
    model.save(path)
    return

def loadisotonicRegression(path):
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
