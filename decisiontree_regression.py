# -*- coding: utf-8 -*-
"""
Created on Thu Jun 28 10:25:02 2018
"""

from pyspark.ml.linalg import Vectors
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml import PipelineModel

from pyspark.ml.regression import DecisionTreeRegressor, DecisionTreeModel

from pyspark.ml.feature import VectorAssembler
from pyspark.ml.feature import VectorIndexer
from pyspark.ml.tuning import (CrossValidator, TrainValidationSplit, ParamGridBuilder)
from pyspark.ml.tuning import (CrossValidatorModel, TrainValidationSplitModel)
from pyspark.ml.evaluation import RegressionEvaluator


sc = SparkContext.getOrCreate()
spark = SparkSession(sc)    
 

#parameter yang akan di-grid untuk diproses ke ML-tuning
grid = { "maxDepth" : [2, 3, 4]}

#Parameter Configuration 
dt_params = {
                     "maxDepth" : 3, 
                     "featuresCol":"features", 
                     "labelCol":"label", 
                     "predictionCol" : "prediction", 
                     "maxBins" : 32, 
                     "minInstancesPerNode" : 1, 
                     "minInfoGain" : 0.0, 
                     "maxMemoryInMB" : 256, 
                     "cacheNodeIds" : False, 
                     "checkpointInterval" : 10, 
                     "impurity" : 'variance', 
                     "seed" : None, 
                     "varianceCol" :None
             }


def converterDF(df, cols, features):
    """
        input : df = dataframe  (per feature per column), 
                cols = variable containing list of feature columns OR 
                       list of feature columns (example : ["a", "b", "c"] )
                features = string (example : "features")
                
        output : dataframe (features in one column)
                 
    """
    converter = VectorAssembler(inputCols=cols,outputCol=features)
    converter_df = converter.transform(df)
    return converter_df


#tuning parameter, metode : Cross Validation (crossval) dan Train Validation Split (trainvalsplit)
#methodParam untuk crossval : int (bilangan bulat)
#method param untuk trainval : pecahan antara 0-1
tune_params = { 
                 "method" : "crossval", 
                 "paramGrids" : grid, 
                 "methodParam" : 3  
                }

#conf1 digunakan apabila tidak akan dilakukan tuning parameter
conf1 = {   
              "params" : dt_params,
              "tuning" : None
        }


#conf2 digunakan apabila akan dilakukan tuning parameter
conf2 = {   
              "params" : dt_params,
              "tuning" : tune_params
        }


#Fungsi untuk mendapatkan model dari data (trained model)
def dtRegression(df, conf):
    """ 
        input : df [spark.dataframe], conf [configuration params]
        output : decisiontree_regression model [model]
    """
    featuresCol = conf["params"].get("featuresCol")
    impurity = conf["params"].get("impurity", "variance")
    
    maxDepth = conf["params"].get("maxDepth", 5)
    maxBin = conf["params"].get("maxBins",32)
    minInstancesPerNode = conf["params"].get("minInstancesPerNode", 1)
    minInfoGain = conf ["params"].get("minInfoGain", 0.0)
    maxMemoryInMB = conf["params"].get("maxMemoryInMB",256)
    cacheNodeIds = conf["params"].get("cacheNodeIds", False)
    checkpointInterval = conf["params"].get("checkpointInterval", 10)
    seed = conf["params"].get("seed", None)
    varianceCol = conf["params"].get("varianceCol", None)   
    
    dt = DecisionTreeRegressor(maxDepth=maxDepth,featuresCol=featuresCol)
    pipeline = Pipeline(stages=[featureIndexer, dt])
    
    print ("maxDepth : " , dt.getMaxDepth())
    
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
            cv = CrossValidator(estimator=pipeline, estimatorParamMaps=grid, 
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
            tvs = TrainValidationSplit(estimator=pipeline, estimatorParamMaps=grid, 
                                       evaluator=evaluator, trainRatio=tr )
            model = tvs.fit(df)
            
    #jika tidak menggunakan ml-tuning
    elif conf["tuning"] == None:
          print ("test")
          model = pipeline.fit(df)
          
    return model


#Menampilkan tree model dari model decision tree (ket : dapat dipanggil apabila tidak menggunakan ML-Tuning)
def dtModel(model):
    """
        input : model (DecisionTreeModel/PipelineModel)
        output : pyspark.ml.regression.DecisionTreeRegressionModel
    """
    
    tModel = model.stages[1]
    return tModel


#Menampilkan validator metri (jika menggunakan ML-Tuning)
def validatorMetrics(model):
    """input : model (TrainValidationSplitModel)
       output : dataframe
    """
    
    vm = model.validationMetrics
    vmet  = []
    for v in vm:
        vmet.append((Vectors.dense(v),))
    df_vm = spark.createDataFrame(vmet, ["Validation Metrics"])
    df_vm.show()
    return df_vm


#Menampilkan average metrics dari CrossValidator Model
def avgMetrics(model):
    """input    : CrossValidatorModel
       output  : dataframe
    """
    avm = model.avgMetrics
    avmet  = []
    for av in avm:
        avmet.append((Vectors.dense(av),))
    df_avm = spark.createDataFrame(avmet, ["Validation Metrics"])
    df_avm.show()
    return df_avm 



#menyimpan model
def saveModel(model, path):
    """
        input : model 
                (CrossValidatorModel / TrainValidationSplitModel / LinearRegressionModel)
        output : void
    """
    
    model.save(path)
    return


#Load Model Decision Tree Regression (jika tidak menggunakan ML-tuning = conf1, jika menggunakan ML-tuning = conf2 )
def loaddtRegression(conf, path):
    """
        input : conf, path
        output : model
                (CrossValidatorModel / TrainValidationSplitModel / PipelineModel)
    """
    
    #Jika menggunakan ML-Tuning
    if conf["tuning"]:    
        #Jika menggunakan Cross Validation, maka tipe model = CrossValidatorModel
        if conf["tuning"].get("method").lower() == "crossval":
            loaded_model = CrossValidatorModel.load(path)        
        #Jika menggunakan Train Validation, maka tipe model = TrainValidationSplitModel   
        elif conf["tuning"].get("method").lower() == "trainvalsplit":
            loaded_model = TrainValidationSplitModel.load(path)
    
    #Jika tidak menggunakan ML-tuning, tipe model = PipelineModel    
    elif conf["tuning"] == None:
        loaded_model = PipelineModel.load(path)
    
    return loaded_model


#menampilkan prediksi test data, jika menggunakan model regresi linear
def predict(df, model):
    """ 
        input   : df [spark.dataframe], linear_regression model [model]
        output  : prediction [dataframe]
    """    
    val = model.transform(df)
    prediction = val.select("label", "prediction")
    return prediction   

    
#menunjukkan nilai R-square 
def summaryR2(df, predictionCol, labelCol):
    """ 
        input : df [spark.dataframe]
        output : R squared on test data [float]
    """    
    lr_evaluator = RegressionEvaluator(predictionCol=predictionCol, 
             labelCol=labelCol, metricName="r2")
    r2 =  lr_evaluator.evaluate(df)
    r2 = [(Vectors.dense(r2),)]
    r2_df = spark.createDataFrame(r2, ["R-square"])
    r2_df.show()
    return r2_df


#menunjukkan nilai rms
def summaryRMSE(df, predictionCol, labelCol):
    """ 
        input : df [spark.dataframe]
        output : RMS on test data [float]
    """    
    lr_evaluator = RegressionEvaluator(predictionCol=predictionCol, 
             labelCol=labelCol, metricName="rmse")
    rmse =  lr_evaluator.evaluate(df)
    rmse = [(Vectors.dense(rmse),)]
    rmse_df = spark.createDataFrame(rmse, ["RMS"])
    rmse_df.show()
    return rmse_df   

     
#memilih hasil pada baris tertentu (prediksi)
def rowSlicing(df, n):
    num_of_data = df.count()
    ls = df.take(num_of_data)
    return ls[n]


#menyalin model
def copyModel(model):
    copy_model = model.copy(extra=None)
    return copy_model





#-----------------------------------------------------------------------------------------------------------



#load input data
#Dataframe input
tree_df = spark.read.format("libsvm").load("C:/Users/Lenovo/spark-master/data/mllib/sample_libsvm_data.txt")


# Automatically identify categorical features, and index them.
# We specify maxCategories so features with > 4 distinct values are treated as continuous.
featureIndexer =\
        VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(tree_df)

training, test = tree_df.randomSplit([0.7, 0.3], seed = 11)



#CONTOH PENGGUNAAN FUNGSI

#getting model
model = dtRegression(training, conf1)

#getting prediction
testing = predict(test, model)

#getting R-square
rsq = summaryR2(testing, "prediction", "label")  

#getting RMS
rms= summaryRMSE(testing, "prediction", "label")
