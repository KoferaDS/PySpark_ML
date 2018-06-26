from pyspark.ml.linalg import Vectors
from pyspark.sql.session import SparkSession

from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.regression import GeneralizedLinearRegressionModel

from pyspark.ml.tuning import (
        CrossValidator, TrainValidationSplit, ParamGridBuilder)
from pyspark.ml.tuning import (CrossValidatorModel, TrainValidationSplitModel)
from pyspark.ml.evaluation import RegressionEvaluator

# set-up spark
spark = SparkSession\
    .builder\
    .appName("GeneralizedLinearRegressionExample")\
    .getOrCreate()

# configurations

# generalized linear regression model params
generalized_params = {
    "labelCol": "label",
    "featuresCol": "features",
    "predictionCol": "prediction",
    "family": "gaussian",
    "link": "identity",
    "fitIntercept": True,
    "maxIter": 10,
    "tol": 1e-6,
    "regParam": 0.3,
    "weightCol": None,
    "solver": "irls",
    "linkPredictionCol": "p",
    "variancePower": 0.0,
    "linkPower": None
}

# used in ML-tuning
grid = {
    "maxIter": [50, 100, 120],
    "regParam": [0.1, 0.01]
}

# tuning params
# methods: Cross Validation ("crossval") and Train Validation Split
#         ("trainvalsplit")
# methodParam: crossval -> integer
#              trainval -> float, range 0.0 - 1.0

tune_params = {
    "method": "crossval",
    "paramGrids": grid,
    "methodParam": 2
}

# used without tuning
conf1 = {
    "params": generalized_params,
    "tuning": None
}

# used with tuning
conf2 = {
    "params": generalized_params,
    "tuning": tune_params
}


# create generalized linear regression from trained data
def generalizedLinearRegressor(dataFrame, conf):
    """
        input: df [spark.dataFrame], conf [configuration params]
        output: generalized linear regression model [model]
    """
    
    # calling params
    label_col = conf["params"].get("labelCol", "label")
    features_col = conf["params"].get("featuresCol", "features")
    prediction_col = conf["params"].get("predictionCol", "prediction")
    fam = conf["params"].get("family", "gaussian")

    fit_intercept = conf["params"].get("fitIntercept", True)
    max_iter = conf["params"].get("maxIter", 25)
    tolp = conf["params"].get("tol", 1e-6)
    reg_param = conf["params"].get("regParam", 0.0)
    weight_col = conf["params"].get("weightCol", None)
    solverp = conf["params"].get("solver", "irls")
    link_prediction_col = conf["params"].get("linkPredictionCol", None)
    variance_power = conf["params"].get("variancePower", 0.0)
    link_power = conf["params"].get("linkPower", None)

    if (fam == "gaussian"):
        li = conf["params"].get("link", "identity")
    elif (fam == "binomial"):
        li = conf["params"].get("link", "logit")
    elif (fam == "poisson"):
        li = conf["params"].get("link", "log")
    elif (fam == "gamma"):
        li = conf["params"].get("link", "inverse")
    elif (fam == "tweedle"):
        li = conf["params"].get("link", 1 - variance_power)
    else:
        li = conf["params"].get("link", None)

    glr = GeneralizedLinearRegression(labelCol=label_col, 
                                      featuresCol=features_col, 
                                      predictionCol=prediction_col, 
                                      family=fam,
                                      link=li, 
                                      fitIntercept=fit_intercept, 
                                      maxIter=max_iter, 
                                      tol=tolp, 
                                      regParam=reg_param, 
                                      solver=solverp, 
                                      linkPredictionCol=link_prediction_col, 
                                      variancePower=variance_power, 
                                      linkPower=link_power)

    # with tuning
    if conf["tuning"]:
        # method: cross validation
        if conf["tuning"].get("method").lower() == "crossval":
            paramGrids = conf["tuning"].get("paramGrids")
            pg = ParamGridBuilder()
            for key in paramGrids:
                pg.addGrid(key, paramGrids[key])

            grid = pg.build()
            folds = conf["tuning"].get("methodParam")
            evaluator = RegressionEvaluator()
            cv = CrossValidator(estimator = glr, estimatorParamMaps = grid, 
                                evaluator = evaluator, numFolds = folds)
            model = cv.fit(dataFrame)

        # method: train validation split
        elif conf["tuning"].get("method").lower() == "trainvalsplit":
            paramGrids = conf["tuning"].get("paramGrids")
            pg = ParamGridBuilder()
            for key in paramGrids:
                pg.addGrid(key, paramGrids[key])

            grid = pg.build()
            tr = conf["tuning"].get("methodParam")
            evaluator = RegressionEvaluator()
            tvs = TrainValidationSplit(estimator = glr, 
                                       estimatorParamMaps = grid, 
                                       evaluator = evaluator, trainRatio = tr)
            model = tvs.fit(dataFrame)

    # without tuning
    else:
        model = glr.fit(dataFrame)

    return model

# show validator metrics (if ML tuning is used)
def validatorMetrics(model):
    """
        input: model [TrainValidationSplitModel]
        output: validation metrics [double]
    """
    return model.validationMetrics

# show average metrics from CrossValidator model
def averageMetrics(model):
    """
        input: model [CrossValidatorModel]
        output: metrics [double]
    """
    return model.avgMetrics

# saving model
def saveModel(model, path):
    """
        input: model [CrossValidatorModel, TrainSplitValidationModel, GeneralizedLinearRegressionModel], path [string]
        output: None
    """
    model.save(path)

# loading model
def loadModel(conf, path):
    """
        input : conf [dictionary], path [string]
        output: model [CrossValidatorModel, TrainValidationSplitModel, GeneralizedLinearRegressionModel]
    """
    if conf["tuning"]:
        if conf["tuning"].get("method").lower() == "crossval":
            loading_model = CrossValidatorModel.load(path)
        elif conf["tuning"].get("method").lower() == "trainvalsplit":
            loading_model = TrainValidationSplitModel.load(path)

    elif conf["tuning"] == None:
        loading_model = GeneralizedLinearRegressionModel.load(path)

    return loading_model

# create prediction from data frame by using generalized linear regression
def predict(dataFrame, model):
    """
        input  : dataFrame [spark.dataFrame], generalized linear regression
                 model [model]
        output : prediction [data frame]
    """
    val = model.transform(dataFrame)
    prediction = val.select("label", "prediction")
    return prediction

# return R-square value
def summaryR2(dataFrame, predictionCol, labelCol):
    """
        input  : dataFrame [spark.dataFrame]
        output : R squared on test data [float]
    """
    glr_evaluator = RegressionEvaluator(
            predictionCol=predictionCol, labelCol=labelCol, metricName="r2")
    r2 = glr_evaluator.evaluate(dataFrame)
    r2 = [(Vectors.dense(r2),)]
    r2_df = spark.createDataFrame(r2, ["R-square"])
    return r2_df

# return RMS value
def summaryRMSE(dataFrame, predictionCol, labelCol):
    """
        input  : dataFrame [spark.dataFrame]
        output : RMS on test data [float]
    """
    glr_evaluator = RegressionEvaluator(
            predictionCol=predictionCol, labelCol=labelCol, metricName="rmse")
    rmse = glr_evaluator.evaluate(dataFrame)
    rmse = [(Vectors.dense(rmse),)]
    rmse_df = spark.createDataFrame(rmse, ["RMS"])
    return rmse_df

# select value from certain row
def rowSlicing(dataFrame, n):
    """
        input  : dataFrame [spark.dataFrame], n [integer]
        output : data [pyspark.sql.types.Row]
    """
    num_of_data = dataFrame.count()
    ls = dataFrame.take(num_of_data)
    return ls[n]


#--------------------------Testing and Example--------------------------#
    
if __name__ == "__main__":
    
    #import data frame from file
    dataset = spark.read.format("libsvm")\
            .load("sample_linear_regression_data.txt")
    
    # split data into training and test with 7:3 ratio
    training, test = dataset.randomSplit([0.7,0.3], seed = 11)
    training.cache()
    
    # create generalized linear regression model
    model = generalizedLinearRegressor(dataset, conf2)
    
    # create data prediction
    testing = predict(test, model)
    
    # show top 10 rows results
    testing.show(10)
    
    # show R-square value
    r2 = summaryR2(testing, "prediction", "label")
    r2.show()
    
    # show RMS value
    rms = summaryRMSE(testing, "prediction", "label")
    rms.show()
    
    # save model into desired path
    saveModel(model, "generalized_linear_regression_model_example")
    
    # load model from desired path
    model2 = loadModel(conf2, "generalized_linear_regression_model_example")
    
    # show top 10 results from loaded model
    testing2 = predict(test, model2)
    testing2.show(10)