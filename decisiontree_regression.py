"""
Created on Mon May 21 13:39:41 2018

"""
from pyspark.ml.linalg import Vectors
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.regression import DecisionTreeModel
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.tuning import (CrossValidator, ParamGridBuilder)
from pyspark.ml.evaluation import RegressionEvaluator

sc = SparkContext.getOrCreate()
spark = SparkSession(sc)


#Decision Tree Model using train-data
def dtree_reg(train_df, conf):
    """ input : df [spark.dataframe], conf [configuration params]
        output : decisiontree_regression model [model]
    """
    maxDepth    = conf["params"].get("maxDepth")
    featuresCol = conf["params"].get("featuresCol")
    
    dt = DecisionTreeRegressor(maxDepth=maxDepth,featuresCol=featuresCol)
    pipeline = Pipeline(stages=[featureIndexer, dt])
    #Cross Validation
    if conf["crossval"].get("crossval") == True:
            grid = ParamGridBuilder().build()
            evaluator = RegressionEvaluator \
            (labelCol="label", predictionCol="prediction", metricName="r2")
            cv = CrossValidator(estimator=dt, estimatorParamMaps=grid, evaluator=evaluator, 
                        parallelism=2)
            dtModel = cv.fit(train_df)
            
    if conf["crossval"].get("crossval") == False:
            dtModel = pipeline.fit(train_df)
       
    return dtModel

#Save Model
def saveModel(model, path):
    model.save(path)
    return

#Load Model
def loadModel(path):
    model = DecisionTreeModel.load(path)
    return model


#Predict test data using trained-model
def predict(test_df, model):
    """ input   : df [spark.dataframe], linear_regression model [model]
        output  : prediction [dataframe]
    """    
    val = model.transform(test_df)
    prediction = val.select("label","prediction")
    return prediction


#R-square function
def r_square(df, col_prediction, col_label):
    """ input : df [spark.dataframe]
        output : R squared on test data [float]
    """    
    dt_evaluator = RegressionEvaluator(predictionCol=col_prediction, 
                 labelCol=col_label, metricName="r2")
    r2 =  dt_evaluator.evaluate(df)
    r2 = [(Vectors.dense(r2),)]
    r2_df = spark.createDataFrame(r2, ["R-square"])
    return r2_df


#Showing RMSE using test data
def rmse(df, col_prediction, col_label):
        """ input : df [spark.dataframe]
            output : RMS on test data [float]
        """    
        lr_evaluator = RegressionEvaluator(predictionCol=col_prediction, 
                 labelCol=col_label, metricName="rmse")
        rmse =  lr_evaluator.evaluate(df)
        rmse = [(Vectors.dense(rmse),)]
        rmse_df = spark.createDataFrame(rmse, ["RMS"])
        return rmse_df

    
#Showing selected row
def row_slicing(df, n):
    num_of_data = df.count()
    ls = df.take(num_of_data)
    return ls[n]




# Loading data (dataframe-format)
df = spark.read.format("libsvm").load("C:/Users/Lenovo/spark-master/data/mllib/sample_libsvm_data.txt")

# Automatically identify categorical features, and index them.
# We specify maxCategories so features with > 4 distinct values are treated as continuous.
featureIndexer =\
        VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(df)

# Split the data into training and test sets (30% held out for testing)
(train_df, test_df) = df.randomSplit([0.7, 0.3])


#Parameter Configuration 
config       =  {
                 "params" : {"maxDepth" : 2, "featuresCol":"features", "labelCol":"label", 
                             "predictionCol" : "prediction", "maxBins" : 32,
                             "minInstancesPerNode" : 1, "minInfoGain" : 0.0,
                             "maxMemoryInMB" : 256, "chacheNodeIds" : False,
                             "checkpointInterval" : 10, "impurity" : "variance",
                             "seed" : None, "varianceCol" : None
                             },
                 "crossval" : {"crossval" : False, "N" : 5, "metricName" : "r2"}
                 }



#getting model
model = dtree_reg(train_df, config)
treeModel = model.stages[1]

#getting prediction
testing = predict (test_df, model)

#getting R-square
rsq = r_square(testing, "prediction", "label")

#getting RMS
rms = rmse(testing, "prediction", "label")
