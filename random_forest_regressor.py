from __future__ import print_function
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator,ParamGridBuilder,TrainValidationSplit
from pyspark.ml.tuning import CrossValidatorModel,TrainValidationSplitModel
from pyspark.ml.regression import RandomForestRegressor
from pyspark.ml.regression import RandomForestRegressionModel
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import VectorAssembler, VectorIndexer
from pyspark.context import SparkContext
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors

sc = SparkContext.getOrCreate()
spark = SparkSession(sc)  
 
# Set parameter and its value for randomforest regression
rfr_params = {
                "featuresCol":"features", "labelCol":"label", 
                "predictionCol" : "prediction", "maxDepth" : 5, "maxBins" : 32,
                "minInstancesPerNode" : 1, "minInfoGain" : 0.0,
                "maxMemoryInMB" : 256, "cacheNodeIds" : False,
                "checkpointInterval" : 10, "impurity" : "variance",
                "subsamplingRate" : 1.0, "seed" : None, "numTrees" : 20,
                "featureSubsetStrategy" : "auto"
             }   

grid = {
        "maxDepth" : [3,4,5],
        "numTrees" : [10,15,20]
        }

# Set params tuning whether use : - method "crossval" or "trainvalsplit"
#                                 - methodParams is set as fold for "crossval" (value : f>0)
#                                   and trainratio for "trainvalsplit" (value: 0<tr<1)
tuning_params = {
                    "method" : "trainvalsplit",
                    "paramGrids" :  grid,
                    "methodparams" : 0.8
                }  
# Set configuration whether use tuning/non-tuning
conf1    = {
                "params" : rfr_params,
                "tuning" : None
          }
conf2   = {
                "params" : rfr_params,
                "tuning" : tuning_params
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
 
    
def randomforestRegression (df,conf):
    """input  : - Dataframe train (df)
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
            paramGrids = conf["tuning"].get("paramGrids")
            pg=ParamGridBuilder()
            for key in paramGrids:
              pg.addGrid(key, paramGrids[key])
            grid = pg.build()
            evaluator = RegressionEvaluator()
            cv = CrossValidator(estimator=rfr, estimatorParamMaps=grid,
                            evaluator=evaluator, numFolds=folds)
            model = cv.fit(df)
        elif conf["tuning"].get("method").lower() == "trainvalsplit":
            tr = conf["tuning"].get("methodParam", 0.8)
            paramGrids = conf["tuning"].get("paramGrids")
            pg=ParamGridBuilder()
            for key in paramGrids:
              pg.addGrid(key, paramGrids[key])
            grid = pg.build()
            evaluator = RegressionEvaluator()
            tvs = TrainValidationSplit(estimator=rfr, estimatorParamMaps=grid,
                                   evaluator=evaluator, trainRatio=tr)
            model = tvs.fit(df)
    elif conf["tuning"] ==  None:
        model = pipeline.fit(df)
    return model

def rfrmodel (model):
    ''' transform pipelinemodel into RandomForestRegressionModel
       Input   : Pipeline Model
       Output  : Random Forest Model
    '''
    rfrmodel = model.stages[1]
    return rfrmodel
    
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
        loaded_model = PipelineModel.load(path)
    return loaded_model


def predict (df, model):
    """ 
        Input  : -Dataframe test(df)
                 -Trained model (model)
        Output : -Dataframe with prediction column (transformed)
    """      
    transformed = model.transform(df).select("label","prediction")
    return transformed
    
def prediction (df,model):
    """ Show dataframe of prediction in kernel
         Input  : -Dataframe test(df)
                  -Trained model (model)
        Output :  -Dataframe display with prediction column (transformed)
    """
    model.transform(df).show()

def summary_R2(df):
    """ Root square value from the model
        input  : -Dataframe prediction (df)
        output : -Dataframe of Root squared (df)
    """    
    rfr_evaluator = RegressionEvaluator(metricName="r2")
    r2        = rfr_evaluator.evaluate(df)
    vr2       = [(Vectors.dense(r2),)]
    df_r2     = spark.createDataFrame(vr2, ["R^2"])
    return df_r2

def summary_Rmse(df):
    """ Root mean square value from the model
        input  : -Dataframe prediction (df)
        output : -Dataframe of Root squared
    """    
    rfr_evaluator = RegressionEvaluator(metricName="rmse")
    rmse      = rfr_evaluator.evaluate(df)
    vrmse     = [(Vectors.dense(rmse),)]
    df_rmse   = spark.createDataFrame(vrmse, ["Rms"])
    return df_rmse

def copyModel(model):
    copy_model = model.copy(extra=None)
    return copy_model
# ------------------------------Test and examples--------------------------------

#     Loads dataframe
df_rfr = spark.read.format("libsvm")\
            .load("D:/spark/data/mllib/sample_libsvm_data.txt") 
            
#    Assemble dataframe features into one column
#    Set list of column name from the dataframe
column = []
#    Set assembled column name
features_name= ["features"]
#   Set converted dataframe
build = convertDF(df_rfr,column,features_name)    

#     Splitting dataframe into dataframe training and test
#     ex: 0.6 (70%) datainput used as df training and 0.4 (30%) used as df test
(df_training, df_test) = df_rfr.randomSplit([0.7, 0.3])

#     Automatically identify categorical features, and index them.
#     Set maxCategories so features with > 4 distinct values are treated as continuous.
featureIndexer =\
        VectorIndexer(inputCol="features", outputCol="indexedFeatures", 
                      maxCategories=4).fit(df_rfr)
 
#     Applied methods to Data
# IR Model
trained_model = randomforestRegression(df_training,conf2)

##Save model
#save = saveModel(trained_model, "path")

##Load model
#load_irmodel = loadIsotonicRegression("path")
#load_cvmodel = loadCrossValidation("path")
#load_tvsmodel = loadTrainValidationSplit("path")

# Prediction
testing = predict(df_test,trained_model)
testing.show()

# Root square
r2      = summary_R2(testing)  
r2.show() 
## Root mean square
rmse    = summary_Rmse(testing)  
rmse.show()
