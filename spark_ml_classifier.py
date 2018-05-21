
from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.classification import LinearSVC
from pyspark.ml.classification import DecisionTreeClassifier

from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.mllib.util import MLUtils


from pyspark.sql import Row
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import StringIndexer

sc = SparkContext.getOrCreate()
#sc = SparkContext('local')
spark = SparkSession(sc)


data_ = MLUtils.loadLibSVMFile(sc, "data/mllib/sample_binary_classification_data.txt")

training = spark.createDataFrame([
    (0, [2,3,4], 1.0),
    (1, [3,1,2], 0.0),
    (2, [4,0,5], 1.0),
    (3, [5,1,3], 0.0),
    (4, [6,7,9], 1.0),
    (5, [7,5,9], 0.0),
    (6, [3,8,5], 1.0),
    (7, [1,3,9], 0.0),
    (8, [10,3,6], 1.0),
    (9, [11,2,1], 0.0),
    (10, [1,2,12], 1.0),
    (11, [1,2,13], 0.0)
], ["id", "features", "label"])

traindata = spark.createDataFrame([
    ([2,3,4], 1.0),
    ([3,1,2], 0.0),
    ([4,0,5], 1.0),
    ([5,1,3], 0.0),
    ([6,7,9], 1.0),
    ([7,5,9], 0.0),
    ([3,8,5], 1.0),
    ([1,3,9], 0.0),
    ([10,3,6], 1.0),
    ([11,2,1], 0.0),
    ([1,2,12], 1.0),
    ([1,2,13], 0.0)
], ["features", "label"])

dataset = spark.createDataFrame(
     [(Vectors.dense([0.0]), 0.0),
      (Vectors.dense([0.4]), 1.0),
      (Vectors.dense([0.5]), 0.0),
      (Vectors.dense([0.6]), 1.0),
      (Vectors.dense([1.0]), 1.0)] * 10,
     ["features", "label"])

# show the DataFrame
# training.show()

columns = ['id', 'dogs', 'cats']
vals = [
     (1, 2, 0),
     (2, 0, 1) ]

# create DataFrame
df = spark.createDataFrame(vals, columns)


data1 = sc.parallelize([
     Row(label=1.0, features=Vectors.dense(1.0, 1.0, 1.0)),
     Row(label=0.0, features=Vectors.dense(1.0, 2.0, 3.0)),
     Row(label=1.0, features=Vectors.dense(2.0, 2.0, 3.0)),
     Row(label=0.0, features=Vectors.dense(4.0, 2.0, 3.0)) ]).toDF()

data2 = sc.parallelize([
     Row(label=1.0, weight=1.0, features=Vectors.dense(0.0, 5.0)),
     Row(label=0.0, weight=2.0, features=Vectors.dense(1.0, 2.0)),
     Row(label=1.0, weight=3.0, features=Vectors.dense(2.0, 1.0)),
     Row(label=0.0, weight=4.0, features=Vectors.dense(3.0, 3.0))]).toDF()

data3 = spark.createDataFrame([
     (1.0, Vectors.dense(1.0)),
     (0.0, Vectors.sparse(1, [], []))], ["label", "features"])



def svc_classifier(df, conf):
  max_iter = conf["params"].get("maxIter")
  reg_param = conf["params"].get("regParam")
  svm = LinearSVC(maxIter=max_iter, regParam=reg_param)
  if conf["tuning"].get("crossval"):
    grid = ParamGridBuilder().addGrid(svm.maxIter, [0, 1]).build()
    evaluator = BinaryClassificationEvaluator()
    cv = CrossValidator(estimator=svm, estimatorParamMaps=grid, evaluator=evaluator)
    model = cv.fit(df)
  else:
    svm = LinearSVC(maxIter=max_iter, regParam=reg_param)
    model = svm.fit(df)
  return model

def logistic_classifier(df, conf):
  max_iter = conf["params"].get("maxIter")
  reg_param = conf["params"].get("regParam")
  elasticNetParam = conf["params"].get("elasticNetParam")
  family = conf["params"].get("family")
  weight = conf["params"].get("weightCol")
  lr = LogisticRegression(maxIter=max_iter, regParam=reg_param, weightCol=weight)
  if conf["tuning"].get("crossval"):
    grid = ParamGridBuilder().addGrid(lr.maxIter, [0, 1]).build()
    evaluator = BinaryClassificationEvaluator()
    cv = CrossValidator(estimator=lr, estimatorParamMaps=grid, evaluator=evaluator)
    model = cv.fit(dataset)
  else:
    mlor = LogisticRegression(regParam=reg_param, weightCol=weight)
    model = mlor.fit(df)
  return model

def decision_tree_classifier(df, conf):
  max_depth = conf["params"].get("maxDepth")
  label_col = conf["params"].get("labelCol")
  dt = DecisionTreeClassifier(maxDepth=max_depth, labelCol=label_col)
  stringIndexer = StringIndexer(inputCol="label", outputCol="indexed")
  si_model = stringIndexer.fit(df)
  td = si_model.transform(df)
  model = dt.fit(td)
  return model




def prediction(df, model):
  result = model.transform(df).head()
  return result.prediction
  

def row_slicing(df, n):
  num_of_data = df.count()
  ls = df.take(num_of_data)
  return ls[n]


def col_slicing(df, col_name):
  sliced = df.select(col_name)
  return sliced

# showing the sliced columns
# >>> col_slicing.show()

# get the number of columns
# >>> len(df.columns)




    
if __name__ == "__main__":
  
  logistic_params = { "maxIter" : 5, "regParam" : 0.01, "elasticNetParam" : 1.0, 
                      "weightCol" : "weight", "family" : "multinomial"
                    }
  
  svc_params = { "maxIter" : 5, "regParam" : 0.01, "elasticNetParam" : 1.0,  
                 "family" : "multinomial"
                }
  
  dt_params = { "maxDepth" : 2, "labelCol" : "indexed" 
    
               }
  
  conf1 = { "params" : svc_params,
            "tuning" : {"crossval" : False, "folds" : 5 }
          }
  
  conf2 = { "params" : logistic_params,
            "tuning" : {"crossval" : False, "folds" : 5 }
          }
  
  conf3 = { "params" : dt_params, 
            "tuning" : {"crossval" : False, "folds" : 5 }
          }
  
  #test0 = sc.parallelize([Row(features=Vectors.dense(-1.0, -1.0, -1.0))]).toDF()
  test0 = spark.createDataFrame([(Vectors.dense(-1.0),)], ["features"])
  #result = model.transform(test0).head()
  #result.prediction
  
  #svc_model = svc_classifier(data1, conf1)
  #print ("model coefficients : ", svc_model.coefficients)
  #print ("model intercept : ", svc_model.intercept)
  
  #logistic_model = logistic_classifier(data2, conf2)
  #print ("model coefficients : ", logistic_model.coefficients)
  #print ("model intercept : ", logistic_model.intercept)
  
  # input data testing harus satu data (satu observasi), misalnya
  #test0 = sc.parallelize([Row(features=Vectors.dense(-1.0, 1.0))]).toDF()
  
  dt_model = decision_tree_classifier(data3, conf3)
  print ("model coefficients : ", dt_model.numNodes)
  print ("model intercept : ", dt_model.depth)
  print ("feature Importances : ", dt_model.featureImportances)
  print ("num of features : ", dt_model.numFeatures)
  print ("num of classes : ", dt_model.numClasses)

  
  
  
  print ("model prediction : ", prediction(test0, dt_model))
  
  
  data_path = "data/mllib/sample_multiclass_classification_data.txt"
  
  mdf = spark.read.format("libsvm").load(data_path)


  
  #sc.stop()
  
"""
  svm = LinearSVC(maxIter=5, regParam=0.01)
  model = svm.fit(data)
  
  print ("model coefficients : ", model.coefficients)
  print ("model intercept : ", model.intercept)
  
  print ("num of classes : ", model.numClasses)
  print ("num of features : ", model.numFeatures)
  
  test0 = sc.parallelize([Row(features=Vectors.dense(-1.0, -1.0, -1.0))]).toDF()
  result = model.transform(test0).head()
  print ("result predictions : ", result.prediction)
  
  temp_path = "/home/markus/Music/models"
  svm_path = temp_path + "/svm"
  svm.save(svm_path)
  svm2 = LinearSVC.load(svm_path)
  
  print ("MaxIter : ", svm2.getMaxIter())
"""
  
  
  
# exec(open("/home/markus/Music/spark_ml_classifier.py").read())
  
