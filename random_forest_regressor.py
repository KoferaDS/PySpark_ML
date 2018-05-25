from __future__ import print_function
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator,ParamGridBuilder
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.regression import RandomForestRegressionModel
from pyspark.context import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorIndexer

sc = SparkContext.getOrCreate()
spark = SparkSession(sc)  
 
def train_RFR (df,conf):
    """input  : - Dataframe train (df)
                - Hyperparameter configuration (conf)
       output : - Random Forest Regression model (model)
    """     
        
    max_depth  = conf["params"].get("maxDepth")
    num_trees  = conf["params"].get("numTrees")
    rfr        = RandomForestRegressor(maxDepth=max_depth, numTrees=num_trees)
    pipeline   = Pipeline(stages=[featureIndexer,rfr])
    # Configure whether use cross validator/not
    if conf["crossval"].get("crossval") == True:
       grid      = ParamGridBuilder().build()
       evaluator = RegressionEvaluator(metricName="r2")
       cv     = CrossValidator(estimator=rfr, estimatorParamMaps=grid, evaluator=evaluator, 
                    parallelism=2) 
       model  = cv.fit(df)
    if conf["crossval"].get("crossval") == False:
       model  = pipeline.fit(df)
    return model

def df_resultModel (model):
    """Input  : -Trained model
       Output : -Dataframe of predictions
    """
    dfmodel = model.predictions
    return dfmodel

def predict (df, model):
    """Associated with the boundaries at the same index, monotone because of isotonic regression. 
        Input  : -Dataframe test(df)
                 -Trained model
        Output : -Dataframe with features and prediction column
    """      
    transformed = model.transform(df).select("features","prediction")
    df_predict  = transformed.show()
    return df_predict

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
    model = RandomForestRegressionModel.load(path)
    return model

def copyModel(model):
    copy_model = model.copy(extra=None)
    return copy_model

def df_boundsModel (model):
    """ Boundaries in increasing order for which predictions are known.
        Input  : -Trained model
        Output : -Dataframe of model boundaries
    """
    df_boundaries = model.boundaries
    return df_boundaries

def root_square(df, prediction, label):
    """ input  : -Dataframe (df)
        output : -Dataframe of Root squared
    """    
    evaluator = RegressionEvaluator(predictionCol="prediction", labelCol = "label", metricName="r2")
    r2        = evaluator.evaluate(df)
    r2        = [(Vectors.dense(r2),)]
    df_r2     = spark.createDataFrame(r2, ["root mean square"])
    df_r2.show()
    return df_r2

def row_slicing(df, n):
    """ input  : -Dataframe (df)
        output : -Dataframe of n selected row
    """     
    num_of_data = df.count()
    rs          = df.take(num_of_data)
    return rs[n]

# ------------------------------Test and examples--------------------------------

     #Loads dataframe
df_rf = spark.read.format("libsvm")\
            .load("D:\Kofera\spark-master\data\mllib\sample_isotonic_regression_libsvm_data.txt")  
            
    # Define configuration
config    = {
                "crossval" : {"crossval" : False, "N" : 10, "metricName" : "r2"},
                "params"   : {"predictionCol" : "prediction",
                              "labelCol" : "label",
                              "maxDepth" : 2,
                              "numTrees" : 3}
            }
     # Automatically identify categorical features, and index them.
     # Specify maxCategories so features with > 4 distinct values are treated as continuous.           
featureIndexer =\
        VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(df_rf)
   
    # Splitting dataframe into dataframe training and test
df_training, df_test = df_rf.randomSplit([0.6, 0.4], seed=11)
df_training.cache()

    # Applied methods to Input data 
trained_model = train_RFR(df_training,config)
prediction    = predict(df_test,trained_model)
#r_square      = root_square(prediction, "prediction", "label")       
#row_sliced    = row_slicing(df_test,10)
#c_model = copyModel(trained_model)
