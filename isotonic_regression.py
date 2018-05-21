from __future__ import print_function
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.tuning import CrossValidator,ParamGridBuilder
from pyspark.ml.regression import IsotonicRegression
#from pyspark.ml.regression import IsotonicRegressionModel
from pyspark.context import SparkContext
from pyspark.sql import SparkSession
#from pyspark.mllib.util import MLUtils

if __name__ == "__main__":
    sc = SparkContext.getOrCreate()
    spark = SparkSession(sc)
    
# Loads dataframe and define configuration
    df = spark.read.format("libsvm")\
        .load("C:/Users/User/Desktop/spark-master/data/mllib/sample_isotonic_regression_libsvm_data.txt")
        config = {"crossval" : {"crossval" : True, "N" : 10, "metricName" : "r2",}}
   
# Splitting between data training and test
    training, test = df.randomSplit([0.6, 0.4], seed=11)
    training.cache()
    
# Trains an isotonic regression model. 
    def Isoton_Regression (df,conf):
        """input :  df [spark.dataframe], config[configuration (Params and use cross validator/not)
           output : Isotonic Regression Model"""
                   
        ir = IsotonicRegression()
        
# Configure whether use cross validator/not
        if conf["crossval"].get("crossval") == True:
            grid = ParamGridBuilder().build()
            evaluator = RegressionEvaluator(metricName="r2")
            cv = CrossValidator(estimator=ir, estimatorParamMaps=grid, evaluator=evaluator, 
                        parallelism=2) 
            irmodel= cv.fit(training)
        if conf["crossval"].get("crossval") == False:
            
            irmodel= ir.fit (training)
            
        return irmodel
    
# Define Isotonic Regression Model        
    ir_model = Isoton_Regression(training,config)
          
# Making prediction using test data  
    def predict (test,model):
        val = ir_model.transform(test)
        val.show()
        return val
        
    testing = predict(test,ir_model)
    
# Showing R-square using test data
    def r_square(col_prediction, col_label):
        """ input : df [spark.dataframe]
            output : R squared on test data [float]
        """    
        ir_evaluator = RegressionEvaluator(predictionCol=col_prediction,
                                           labelCol=col_label, metricName="r2")
        r2 =  ir_evaluator.evaluate(testing)
        return r2
    
    
    rsq = r_square("prediction","label")
    print()
    
# Showing selected row
    def row_slicing(df, n):
        num_of_data = df.count()
        ls = df.take(num_of_data)
        return ls[n]
 
    spark.stop()
