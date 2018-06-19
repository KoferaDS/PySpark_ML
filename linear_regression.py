from pyspark.ml.linalg import Vectors
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession

from pyspark.ml.regression import LinearRegression
from pyspark.ml.regression import LinearRegressionModel

from pyspark.ml.tuning import (CrossValidator, TrainValidationSplit, ParamGridBuilder)
from pyspark.ml.tuning import (CrossValidatorModel, TrainValidationSplitModel)
from pyspark.ml.evaluation import RegressionEvaluator


sc = SparkContext.getOrCreate()
spark = SparkSession(sc)    
 
 
#hiperparameter yang akan di grid (kemudian diproses pada ML-tuning)
grid = { "maxIter" : [50, 100, 120], "regParam" : [0.1, 0.01]}


#parameter regresi linear
linear_params = { 
                    "regParam" : 0.01, 
                    "elasticNetParam" : 1.0, 
                    "maxIter" : 100, 
                    "tol" : 1e-06, "fitIntercept" : True, 
                    "standardization" : True, 
                    "solver" : "auto", 
                    "weightCol" : None, 
                    "aggregationDepth" : 2, 
                    "loss" : "squaredError", 
                    "epsilon" : 1.35
                }

#tuning parameter, metode : Cross Validation (crossval) dan Train Validation Split (trainvalsplit)
#methodParam untuk crossval : int (bilangan bulat) contoh : 3
#method param untuk trainval : pecahan antara 0 sampai 1
tune_params = { 
                 "method" : "trainvalsplit", 
                 "paramGrids" : grid, 
                 "methodParam" : 0.8  
                }

#conf1 digunakan apabila tidak akan dilakukan tuning parameter
conf1 = {   
          "params" : linear_params,
          "tuning" : None
        }


#conf2 digunakan apabila akan dilakukan tuning parameter
conf2 = {   
          "params" : linear_params,
          "tuning" : tune_params
        }


#Membuat model menggunakan regresi linear (dari data training)
def linearRegressor(df, conf):
    """ 
        input : df [spark.dataframe], conf [configuration params]
        output : linear_regression model [model]
    """
    #memanggil parameter (nilai default)
    featuresCol= conf["params"].get("featuresCol", "features")
    labelCol= conf["params"].get("labelCol", "label")
    predictionCol = conf["params"].get("predictionCol", "prediction")
        
    max_iter = conf["params"].get("maxIter", 100)
    reg_param = conf["params"].get("regParam", 0.0)
    elasticnet_param = conf["params"].get("elasticNetParam", 0.0)
    tol = conf["params"].get("tol", 1e-6)
    fitIntercept = conf["params"].get("fitIntercept", True)
    standardization = conf["params"].get("standardization", True)
    solver = conf["params"].get("solver", "auto")
    weightCol = conf["params"].get("weightCol", None)
    aggregationDepth = conf["params"].get("aggregationDepth", 2)
    loss = conf["params"].get("loss", "squaredError")
    epsilon =  conf["params"].get("epsilon", 1.35)        
        
    lr = LinearRegression(maxIter=max_iter, regParam=reg_param, 
                              elasticNetParam=elasticnet_param)
        
    print ("maxIter : " , lr.getMaxIter())
    print ("regParam : " , lr.getRegParam())
    print ("aggrDepth : " , lr.getAggregationDepth())
        
    #jika menggunakan ml-tuning
    if conf["tuning"]:
            
        #jika menggunakan ml-tuning cross validation  
        if conf["tuning"].get("method").lower() == "crossval":
            paramgGrids = conf["tuning"].get("paramGrids")
            pg = ParamGridBuilder()
            for key in paramgGrids:
              pg.addGrid(key, paramgGrids[key])
          
            grid = pg.build()
            folds = conf["tuning"].get("methodParam")
            evaluator = RegressionEvaluator()
            cv = CrossValidator(estimator=lr, estimatorParamMaps=grid, 
                                evaluator=evaluator, numFolds= folds)
            model = cv.fit(df)
          
        #jika menggunakan ml-tuning train validation split
        elif conf["tuning"].get("method").lower() == "trainvalsplit":
            paramgGrids = conf["tuning"].get("paramGrids")
            pg = ParamGridBuilder()
            for key in paramgGrids:
              pg.addGrid(key, paramgGrids[key])
          
            grid = pg.build()
            tr = conf["tuning"].get("methodParam")
            evaluator = RegressionEvaluator()
            tvs = TrainValidationSplit(estimator=lr, estimatorParamMaps=grid, 
                                       evaluator=evaluator, trainRatio=tr )
            model = tvs.fit(df)
            
    #jika tidak menggunakan ml-tuning
    elif conf["tuning"] == None:
          print ("test")
          model = lr.fit(df)
          
    return model

#Menampilkan validator metri (jika menggunakan ML-Tuning)
def validatorMetrics(model):
    """
       input : model (TrainValidationSplitModel)
       output : validation metrics 
    """
    vm = model.validationMetrics
    return vm


#Menampilkan average metrics dari CrossValidator Model
def avgMetrics(model):
    """
       input   : CrossValidatorModel
       output  : metrics
    """
    avm = model.avgMetrics
    return avm   


#menyimpan model
def saveModel(model, path):
    """
    input  : model (CrossValidatorModel / TrainValidationSplitModel / LinearRegressionModel)
    output : None
    """   
    model.save(path)
 


#Load Model (jika tidak menggunakan ML-tuning = conf1, jika menggunakan ML-tuning = conf2 )
def loadModelLinearRegression(conf, path):
    """
       input  : conf, path
       output : model (CrossValidatorModel / TrainValidationSplitModel / LinearRegressionModel)
    """
                   
    #Jika menggunakan ML-Tuning
    if conf["tuning"]:    
        #Jika menggunakan Cross Validation, maka tipe model = CrossValidatorModel
        if conf["tuning"].get("method").lower() == "crossval":
            load_model = CrossValidatorModel.load(path)        
        #Jika menggunakan Train Validation, maka tipe model = TrainValidationSplitModel   
        elif conf["tuning"].get("method").lower() == "trainvalsplit":
            load_model = TrainValidationSplitModel.load(path)
    
    #Jika tidak menggunakan ML-tuning, tipe model = LinearRegressionModel    
    elif conf["tuning"] == None:
        load_model = LinearRegressionModel.load(path)
    
    return load_model


#menampilkan prediksi test data, jika menggunakan model regresi linear
def predict(df, model):
    """ input  : df [spark.dataframe], linear_regression model [model]
        output : prediction [dataframe]
    """    
    val = model.transform(df)
    prediction = val.select("label", "prediction")
    return prediction
    
    
#menunjukkan nilai R-square 
def summaryR2(df, predictionCol, labelCol):
    """ 
        input  : df [spark.dataframe]
        output : R squared on test data [float]
    """    
    lr_evaluator = RegressionEvaluator(predictionCol=predictionCol, 
             labelCol=labelCol, metricName="r2")
    r2 =  lr_evaluator.evaluate(df)
    r2 = [(Vectors.dense(r2),)]
    r2_df = spark.createDataFrame(r2, ["R-square"])
    return r2_df


#menunjukkan nilai rms
def summaryRMSE(df, predictionCol, labelCol):
    """ 
        input  : df [spark.dataframe]
        output : RMS on test data [float]
    """    
    lr_evaluator = RegressionEvaluator(predictionCol=predictionCol, 
             labelCol=labelCol, metricName="rmse")
    rmse =  lr_evaluator.evaluate(df)
    rmse = [(Vectors.dense(rmse),)]
    rmse_df = spark.createDataFrame(rmse, ["RMS"])
    return rmse_df 
  
    
#memilih hasil pada baris tertentu (prediksi)
def rowSlicing(df, n):
    """
    df : spark-dataFrame
    n  : row index
    """
    num_of_data = df.count()
    ls = df.take(num_of_data)
    return ls[n]








#--------------------------Test dan Contoh Penggunaan--------------------------#
#load input data
linear_df = spark.read.format("libsvm")\
        .load("C:/Users/Lenovo/spark-master/data/mllib/sample_linear_regression_data.txt")


#Splitting data into training and test
training, test = linear_df.randomSplit([0.7, 0.3], seed = 11)       
training.cache()                                 
                
#mendapatkan model
model = linearRegressor(training, conf1)

#melakukan prediksi menggunakan data test
testing = predict(test, model)
testing.show(10)

#mencari dan menampilkan R-square dari data prediksi
r2 = summaryR2(testing, "prediction", "label")
r2.show()

#mencari dan menampilkan nilai RMS dari data prediksi
rms= summaryRMSE(testing, "prediction", "label")
rms.show()
