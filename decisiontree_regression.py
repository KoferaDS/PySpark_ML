# -*- coding: utf-8 -*-
from pyspark.ml.linalg import Vectors
from pyspark.context import SparkContext
from pyspark.sql.session import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml import PipelineModel

from pyspark.ml.regression import DecisionTreeRegressor
from pyspark.ml.regression import DecisionTreeModel

from pyspark.ml.feature import VectorIndexer
from pyspark.ml.tuning import (TrainValidationSplit, CrossValidator, ParamGridBuilder)
from pyspark.ml.tuning import (CrossValidatorModel, TrainValidationSplitModel)
from pyspark.ml.evaluation import RegressionEvaluator

sc = SparkContext.getOrCreate()
spark = SparkSession(sc)

#Parameter Configuration 
config       =  {
                 "params" : {"maxDepth" : 3, "featuresCol":"features", "labelCol":"label", 
                             "predictionCol" : "prediction"},
                             
                 #tuning method = None, jika tidak menggunakan ML-Tuning
                 #tuning method = crossval, jika menggunakan ML-Tuning Cross Validation
                 #tuning method = trainval, jika menggunakan ML-Tuning Train Validation Split
                 "tuning" : {"method" : "trainval" , "methodParam" : 0.8}
                 }



#Fungsi untuk mendapatkan model dari data (trained model)
def dtRegressor(df, conf):
    """ input : df [spark.dataframe], conf [configuration params]
        output : decisiontree_regression model [model]
    """
    featuresCol = conf["params"].get("featuresCol")
    impurity = conf["params"].get("impurity", "variance")
    
    maxDepth    = conf["params"].get("maxDepth", 5)
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
    
    #Jika menggunakan ML-Tuning Cross Validation
    if conf["tuning"].get("method") == "crossval":
            grid = ParamGridBuilder().addGrid(dt.maxDepth, [3,4,5]).build()
            fold = conf["tuning"].get("methodParam")
            evaluator = RegressionEvaluator \
            (labelCol="label", predictionCol="prediction", metricName="r2")
            cv = CrossValidator(estimator=pipeline, estimatorParamMaps=grid, evaluator=evaluator, 
                        numFolds=fold)
            model = cv.fit(df)
    
    #Jika menggunakan ML-Tuning Train Validation Split
    elif conf["tuning"].get("method") == "trainval":
            grid = ParamGridBuilder().addGrid(dt.maxDepth, [3,4,5]).build()
            tr = conf["tuning"].get("methodParam")
            evaluator = RegressionEvaluator()
            tvs = TrainValidationSplit(estimator=pipeline, estimatorParamMaps=grid, 
                                       evaluator=evaluator, trainRatio=tr )
            model = tvs.fit(df)
    
    #Jika tidak menggunakan ML-Tuning        
    elif conf["tuning"].get("method") == None:
            model = pipeline.fit(df)
       
    return model

#Menampilkan tree model (ket : dapat dipanggil apabila tidak menggunakan ML-Tuning)
def treeModel(model):
    tModel = model.stages[1]
    return tModel    

#Menampilkan validator metri (jika menggunakan ML-Tuning)
def validatorMetrics(model):
    vm = model.validationMetrics
    return vm

#Save Model
def saveModel(model, path):
    model.save(path)

#Load Model
def loadModel(path): 
    #Jika menggunakan ML-Tuning Cross Validation, maka tipe model = CrossValidatorModel
    if config["tuning"].get("method") == "crossval" :
        model = CrossValidatorModel.load(path)        
    #Jika menggunakan ML-Tuning Train Validation, maka tipe model = TrainValidationSplitModel
    elif config["tuning"].get("method") == "trainval":
        model = TrainValidationSplitModel.load(path)
    #Jika tidak menggunakan ML-Tuning, maka tipe model = DecisionTreeModel    
    elif config["tuning"].get("method") == None:
        model = PipelineModel.load(path)
        
    return model


#Fungsi untuk melakukan prediksi dengan menggunakan trained model
def predict(df, model):
    """ input   : df [spark.dataframe], linear_regression model [model]
        output  : prediction [dataframe]
    """    
    val = model.transform(df)
    prediction = val.select("label","prediction")
    return prediction


#funsi untuk mendapat nilai R-square
def summaryR2(df, col_prediction, col_label):
    """ input : df [spark.dataframe]
        output : R squared on test data [float]
    """    
    dt_evaluator = RegressionEvaluator(predictionCol=col_prediction, 
                 labelCol=col_label, metricName="r2")
    r2 =  dt_evaluator.evaluate(df)
    r2 = [(Vectors.dense(r2),)]
    r2_df = spark.createDataFrame(r2, ["R-square"])
    return r2_df


#fungsi untuk mendapat nilai RMS
def summaryRMSE(df, col_prediction, col_label):
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



#--------------------------Test dan Contoh Penggunaan--------------------------#

#Dataframe input
df = spark.read.format("libsvm").load("C:/Users/Lenovo/spark-master/data/mllib/sample_libsvm_data.txt")

# Automatically identify categorical features, and index them.
# We specify maxCategories so features with > 4 distinct values are treated as continuous.
featureIndexer =\
        VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=4).fit(df)

#Mencacah dataframe menjadi train dan test data dengan ratio 70% train data, 30% test data
(train, test) = df.randomSplit([0.7, 0.3])


#mendapatkan model menggunakan fungsi dtRegressor, dengan train data.
model = dtRegressor(train, config)

#mendapatkan dan menampilkan prediksi dari test data dengan menggunakan model yang sudah di-train
testing = predict (test, model)
testing.show(10)

#menampilkan R-square dari hasil prediksi
r2 = summaryR2(testing, "prediction", "label")
r2.show()

#menampilkan RMS dari hasil prediksi
rms = summaryRMSE(testing, "prediction", "label")
rms.show()
