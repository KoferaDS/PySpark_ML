from __future__ import print_function

from pyspark.ml.regression import LinearRegression
from pyspark.ml.tuning import (CrossValidator, ParamGridBuilder)
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.ml.regression import DecisionTreeRegressor
#import tempfile

if __name__ == "__main__":
    sc = SparkContext.getOrCreate()
    spark = SparkSession(sc)
#    temp_path=tempfile.mkdtemp()
    
    
    # Load training data
    df = spark.read.format("libsvm")\
        .load("C:/Users/Lenovo/spark-master/data/mllib/sample_linear_regression_data.txt")
    
    #Config Dictionary
    glm_params = {"maxIter" : 50, "regParam" : 0.1, "elasticNetParam":0.5}
    dtree_params = { }
    config={"params" : glm_params,
            "crossval" : {"crossval" : False, "N" : 5, "metricName" : "r2"}
            }
    
    #Splitting data into training and test
    training, test = df.randomSplit([0.6, 0.4], seed=11)
    training.cache()
    
    
    #Making Linear Regression Model using training data
    def linear_reg(df, conf):
        """ input : df [spark.dataframe], conf [configuration params]
            output : linear_regression model [model]
        """
        max_iter = conf["params"].get("maxIter")
        reg_param = conf["params"].get("regParam")
        elasticnet_param = conf["params"].get("elasticNetParam")
        
        lr = LinearRegression(maxIter=max_iter, regParam=reg_param, elasticNetParam=elasticnet_param)
        
        #Cross Validation
        if conf["crossval"].get("crossval") == True:
            grid = ParamGridBuilder().build()
            evaluator = RegressionEvaluator(metricName="r2")
            cv = CrossValidator(estimator=lr, estimatorParamMaps=grid, evaluator=evaluator, 
                        parallelism=2)
            lrModel = cv.fit(training)
            
        if conf["crossval"].get("crossval") == False:
            lrModel = lr.fit(training)
            
        return lrModel
    
    #Decision Tree Regression
#    def dtree_reg(df,conf):
    
    model = linear_reg(training, config)
    
#    def savemodel(df, path):
#        model_path = temp_path + "/lr"
#        lr.save(model_path)
    
#    saved_model=savemodel(training, lr)    
    
#    def loadmodel(df,model):
#        load_model = LinearRegressionModel.load(saved_model)
#        return load_model
    
    #Making Prediction using test data
    def predict(test, model):
        """ input   : df [spark.dataframe], linear_regression model [model]
            output  : prediction
        """    
        val = model.transform(test)
        prediction = val.select("label","prediction")
        return prediction
    
    testing = predict(test, model)
    
#    Showing R-square using test data
    def r_square(col_prediction, col_label):
        """ input : df [spark.dataframe]
            output : R squared on test data [float]
        """    
        lr_evaluator = RegressionEvaluator(predictionCol=col_prediction, 
                 labelCol=col_label, metricName="r2")
        r2 =  lr_evaluator.evaluate(testing)
        return r2
    
    
    rsq = r_square("prediction","label")
    
#    Showing selected row
    def row_slicing(df, n):
        num_of_data = df.count()
        ls = df.take(num_of_data)
        return ls[n]

spark.stop()