[ 
# Regression Params

 { afts_params = 
  {
  "name" : "AFTSurvivalRegression"
  "params" :[{
               "name" : "predictionCol"
               "type" : "str"
	       "default value": "prediction" 
             },
             {
               "name" : "labelCol"
               "type" : "str"
	       "default value": "label"
             },
             { 
               "name" : "featuresCol"
               "type" : "str"
	       "default value": "features"
             },
             { 
               "name" : "censorCol"
               "type" : "str"
	       "default value": "censor"
             },
             { 
               "name" : "quantilesCol"
               "type" : "str"
	       "default value": None
             },
             { 
               "name" : "fit"int"ercept"
               "type" : "boolean"
	       "default value": True
             },
             { 
               "name" : "maxIter"
               "type" : "int"
	       "default value": 100
             },
             { 
               "name" : "tol"
               "type" : -
	       "default value": 1E-6
             },	
             {
               "name" : "quantileProbabilities"
               "type" : "list"
	       "default value": [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
             },
             {
               "name" : "aggregationDepth"
               "type" : "int"
	       "default value": 2
             }
	    ]
  }
 }
 { linear_params =
  {
  "name" : "LinearRegression"
  "params" : [{
                "name" : "maxIter"
                "type" : "int"
	        "default value": 5
              },
              {
                "name" : "regParam"
                "type" : "float"
	        "default value": 0.01
              },  
              {
                "name" : "elasticNetParam"
                "type" : "float"
	        "default value": 1.0
              },
              {
                "name" : "tol"
                "type" : -
	        "default value": 1e-06
              },
              {
                "name" : "fit"int"ercept"
                "type" : "boolean"
	        "default value": True
              },
              {
                "name" : "standardization"
                "type" : "boolean"
	        "default value": True
              },
              {
                "name" : "solver"
                "type" : -
	        "default value": "auto"
              }, 
              {
                "name" : "weightCol"
                "type" : "str"
	        "default value": "weight"
              }, 
              {
                "name" : "aggregationDepth"
                "type" : "int"
	        "default value": 2
              },
              {
                "name" : "loss"
                "type" : -
	        "default value": "squaredError"
              }, 
              {
                "name" : "epsilon"
                "type" : "float"
	        "default value": 1.35
              }
             ]
  }
 }
 { isotonic_params =
  {
  "name" : "IsotonicRegression"
  "params" : [{
                "name" : "predictionCol"
                "type" : "str"
	        "default value": "prediction"
              },
              {
                "name" : "labelCol"
                "type" : "str"
	        "default value": "label"
              },  
              {
                "name" : "featuresCol"
                "type" : "str"
	        "default value": "features"
              },
              {
                "name" : "weightCol"
                "type" : "str"
	        "default value": "weight"
              },
              {
                "name" : "isotonic"
                "type" : "boolean"
	        "default value": True
              },
              {
                "name" : "featureIndex"
                "type" : "int"
	        "default value": 0
              }
	     ]
  }
 }
 { dt_params =
  { 
   "name" : "DecisiontreeRegression"
   "params" : [{
                 "name" : "maxDepth"
                 "type" : "int"
	         "default value": 3
               },
               {
                 "name" : "featuresCol"
                 "type" : "str"
	         "default value": "features"
               },  
               {
                 "name" : "labelCol"
                 "type" : "str"
	         "default value": "label"
               },
               {
                 "name" : "predictionCol"
                 "type" : "str"
	         "default value": "prediction"
               },
               {
                 "name" : "maxBins"
                 "type" : "int"
	         "default value": 32
               },
               {
                 "name" : "minInstancesPerNode"
                 "type" : "int"
	         "default value": 1
               },
               {
                 "name" : "minInfoGain"
                 "type" : "float"
	         "default value": 0.0
               }, 
               {
                 "name" : "maxMemoryInMB"
                 "type" : "int"
	         "default value": 256
               }, 
               {
                 "name" : "cacheModeIds"
                 "type" : "boolean"
	         "default value": False
               }, 
               {
                 "name" : "checkpo"int""int"erval"
                 "type" : "int"
	         "default value": 10
               },
               {
                 "name" : "impurity"
                 "type" : -
	         "default value": "variance"
               }, 
               {
                 "name" : "seed"
                 "type" : "int"
	         "default value": None
               },
               {
                 "name" : "varianceCol"
                 "type" : "str"
	         "default value": None
               }
	      ]
  }
 }
 { rfr_params =
  {
   "name" : "RandomforestRegression"
   "params" : [{
                 "name" : "featuresCol"
                 "type" : "str"
	         "default value": "features"
               },  
               {
                 "name" : "labelCol"
                 "type" : "str"
	         "default value": "label"
               },
               {
                 "name" : "predictionCol"
                 "type" : "str"
	         "default value": "prediction"
               },
               {
                 "name" : "maxDepth"
                 "type" : "int"
	         "default value": 5
               },
               {
                 "name" : "maxBins"
                 "type" : "int"
	         "default value": 32
               },
               {
                 "name" : "minInstancesPerNode"
                 "type" : "int"
	         "default value": 1
               },
               {
                 "name" : "minInfoGain"
                 "type" : "float"
	         "default value": 0.0
               }, 
               {
                 "name" : "maxMemoryInMB"
                 "type" : "int"
	         "default value": 256
               }, 
               {
                 "name" : "cacheNodeIds"
                 "type" : "boolean"
	         "default value": False
               }, 
               {
                 "name" : "checkpo"int""int"erval"
                 "type" : "int"
	         "default value": 10
               },
               {
                 "name" : "impurity"
                 "type" : -
	         "default value": "variance"
               }, 
               {
                 "name" : "subsamplingRate"
                 "type" : "float"
	         "default value": 1.0
               },
               {
                 "name" : "seed"
                 "type" : "int"
	         "default value": None
               },
               {
                 "name" : "numTrees"
                 "type" : "int"
	         "default value": 20
               },
               {
                 "name" : "featureSubsetStrategy"
                 "type" : -
	         "default value": "auto"
               }
              ]
  }
 }
 { gbt_params =
  {
   "name" : GradientboostedtreesRegression" 
   "params" :[{
                "name" : "maxIter"
                "type" : "int"
	        "default value": 20
              },  
              {
                "name" : "maxDepth"
                "type" : "int"
	        "default value": 3
              },
              {
                "name" : "featuresCol"
                "type" : "str"
	        "default value": "features"
              },
              {
                "name" : "labelCol"
                "type" : "str"
	        "default value": "label"
              },
              {
                "name" : "predictionCol"
                "type" : "str"
	        "default value": "prediction"
              },
              {
                "name" : "maxBins"
                "type" : "int"
	        "default value": 32
              },
              {
                "name" : "minInstancesPerNode"
                "type" : "int"
	        "default value": 1
              },
              {
                "name" : "minInfoGain"
                "type" : "float"
	        "default value": 0.0
              }, 
              {
                "name" : "maxMemoryInMB"
                "type" : "int"
	        "default value": 256
              }, 
              {
                "name" : "cacheNodeIds"
                "type" : "boolean"
	        "default value": False
              }, 
              {
                "name" : "subsamplingRate"
                "type" : "float"
	        "default value": 1.0
              },
              {
                "name" : "checkpo"int""int"erval"
                "type" : "int"
	        "default value": 10
              },
              {
                "name" : "lossType"
                "type" : -
	        "default value": "squared"
              },
              {
                "name" : "stepSize"
                "type" : "float"
	            "default value": 0.1
              },
              {
                "name" : "seed"
                "type" : "int"
	            "default value": None
              },
              {
                "name" : "impurity"
                "type" : -
	            "default value": "variance"
              } 
             ]
  }
 }
]
