from pyspark.ml.linalg import Vectors
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import LinearRegressionModel
from pyspark.ml.tuning import (CrossValidator, ParamGridBuilder)
from pyspark.ml.evaluation import RegressionEvaluator

sc = SparkContext.getOrCreate()
spark = SparkSession(sc)    
 
   
#Making Linear Regression Model using training data
def linear_reg(train_df, conf):
        """ input : df [spark.dataframe], conf [configuration params]
            output : linear_regression model [model]
        """
        max_iter = conf["params"].get("maxIter")
        reg_param = conf["params"].get("regParam")
        elasticnet_param = conf["params"].get("elasticNetParam")
        featuresCol= conf["params"].get("featuresCol")
        labelCol= conf["params"].get("labelCol")
        predictionCol = conf["params"].get("predictionCol")
        tol = conf["params"].get("tol")
        fitIntercept = conf["params"].get("fitIntercept")
        standardization = conf["params"].get("standardization")
        solver = conf["params"].get("solver")
        weightCol = conf["params"].get("weightCol")
        aggregationDepth = conf["params"].get("aggregationDepth")
        loss = conf["params"].get("loss")
        epsilon =  conf["params"].get("epsilon")
        
        lr = LinearRegression(maxIter=max_iter, regParam=reg_param, elasticNetParam=elasticnet_param)
        
        #Cross Validation
        if conf["crossval"].get("crossval") == True:
            grid = ParamGridBuilder().build()
            evaluator = RegressionEvaluator(metricName="r2")
            cv = CrossValidator(estimator=lr, estimatorParamMaps=grid, evaluator=evaluator, 
                        parallelism=2)
            lrModel = cv.fit(train_df)
            
        if conf["crossval"].get("crossval") == False:
            lrModel = lr.fit(train_df)
            
        return lrModel

#Save Model
def saveModel(model, path):
    model.save(path)
    return

#Load Model
def loadModel(path):
    model = LinearRegressionModel.load(path)
    return model

#Making Prediction using test data
def predict(test_df, model):
        """ input   : df [spark.dataframe], linear_regression model [model]
            output  : prediction [dataframe]
        """    
        val = model.transform(test_df)
        prediction = val.select("label","prediction")
        return prediction
    
#Showing R-square using test data
def r_square(df, col_prediction, col_label):
        """ input : df [spark.dataframe]
            output : R squared on test data [float]
        """    
        lr_evaluator = RegressionEvaluator(predictionCol=col_prediction, 
                 labelCol=col_label, metricName="r2")
        r2 =  lr_evaluator.evaluate(df)
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




#load input data
linear_df = spark.read.format("libsvm")\
        .load("C:/Users/Lenovo/spark-master/data/mllib/sample_linear_regression_data.txt")

#Splitting data into training and test
training, test = linear_df.randomSplit([0.7, 0.3])       
training.cache()
    
#Config Dictionary
config      = {
               "params" : {"maxIter" : 10, "regParam" : 0.3, "elasticNetParam": 0.8,
                           "featuresCol":"features", "labelCol":"label", 
                            "predictionCol":"prediction", "tol": 1e-06, 
                            "fitIntercept":True, "standardization":True, 
                            "solver":"auto", "weightCol":None, "aggregationDepth":2, 
                            "loss":"squaredError", "epsilon":1.35 
                           },
               "crossval" : {"crossval" : False, "N" : 10, "metricName" : "r2"}
               }   

#getting model
model = linear_reg(training, config)

#getting prediction
testing = predict(test, model)

#getting R-square
rsq = r_square(testing, "prediction", "label")  

#getting RMS
rms= rmse(testing, "prediction", "label")
