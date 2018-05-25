from __future__ import print_function
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator,ParamGridBuilder
from pyspark.ml.regression import IsotonicRegression
from pyspark.ml.regression import IsotonicRegressionModel
from pyspark.context import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors

sc = SparkContext.getOrCreate()
spark = SparkSession(sc)  
 
    
def train_IR (df,conf):
    """input  : - Dataframe train (df)
                - Hyperparameter configuration (conf)
       output : - Isotonic Regression model (model)
    """     
    ir        = IsotonicRegression() 
   
    # Configure whether use cross validator/not
    if conf["crossval"].get("crossval") == True:
       grid      = ParamGridBuilder().build()
       evaluator = RegressionEvaluator(metricName="r2")
       cv     = CrossValidator(estimator=ir, estimatorParamMaps=grid, evaluator=evaluator, 
                    parallelism=2) 
       model  = cv.fit(df)
    if conf["crossval"].get("crossval") == False:
       model  = ir.fit(df)
    return model

def saveModel(model,path):
    '''Save model into corresponding path.
       Input  : -model
                -path
       Output : -saved model
    '''
    model.save(path)
    return

def loadModel(path):
    '''Loading model from path.
       Input  : -Path
       Output : -Loaded model
    '''
    model = IsotonicRegressionModel.load(path)
    return model


def predict (df, model):
    """Associated with the boundaries at the same index, monotone because of isotonic regression. 
        Input  : -Dataframe test(df)
                 -Trained model (model)
        Output : -Dataframe with label, features, and prediction column
    """      
    transformed = model.transform(df).select("label","features","prediction")
    return transformed

def Rsquare(df, prediction, label):
    """ input  : -Dataframe prediction (df)
        output : -Dataframe of Root squared (df)
    """    
    ir_evaluator = RegressionEvaluator(predictionCol="prediction", 
                                       labelCol = "label", metricName="r2")
    r2        = ir_evaluator.evaluate(df)
    vr2       = [(Vectors.dense(r2),)]
    df_r2     = spark.createDataFrame(vr2, ["R^2"])
    return df_r2

def Rmse(df, prediction, label):
    """ input  : -Dataframe prediction (df)
        output : -Dataframe of Root squared
    """    
    ir_evaluator = RegressionEvaluator(labelCol = "label",predictionCol="prediction", 
                                       metricName="rmse")
    rmse      = ir_evaluator.evaluate(df)
    vrmse     = [(Vectors.dense(rmse),)]
    df_rmse   = spark.createDataFrame(vrmse, ["Rms"])
    return df_rmse

def copyModel(model):
    copy_model = model.copy(extra=None)
    return copy_model
# ------------------------------Test and examples--------------------------------

#     Loads dataframe
df_isoton = spark.read.format("libsvm")\
            .load("D:\Kofera\spark-master\data\mllib\sample_isotonic_regression_libsvm_data.txt")    
    
    # Splitting dataframe into dataframe training and test
df_training, df_test = df_isoton.randomSplit([0.6, 0.4], seed=11)
df_training.cache()

#     Define params and config      
ir_params = {
                "predictionCol" : "prediction",
                "labelCol" : "label"
            }            
config    = {
                "params" : ir_params,
                "crossval" : {"crossval" : False, "N" : 5, "metricName" : "r2"},
            }
   
    # Applied methods to Data
# IR Model
trained_model = train_IR(df_training,config)
# Prediction
testing = predict(df_test,trained_model)
# Select row to display
row_sliced    = testing.show(5)
# Root square
r2      = Rsquare(testing, "prediction", "label")  
r2.show() 
# Root mean square
rmse    = Rmse(testing,"prediction", "label")  
rmse.show()
