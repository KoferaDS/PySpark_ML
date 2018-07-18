[ 
# Regression Params

 { afts_params = 
  {
   "name" : "AFTSurvivalRegression"
   "params" :[{
               "name" : "predictionCol"
               "type" : str
	       "default": "prediction" 
             },
             {
               "name" : "labelCol"
               "type" : str
	       "default": "label"
             },
             { 
               "name" : "featuresCol"
               "type" : str
	       "default": "features"
             },
             { 
               "name" : "censorCol"
               "type" : str
	       "default": "censor"
             },
             { 
               "name" : "quantilesCol"
               "type" : str
	       "default": None
             },
             { 
               "name" : "fitintercept"
               "type" : bool
	       "default": True
             },
             { 
               "name" : "maxIter"
               "type" : int
	       "default": 100
             },
             { 
               "name" : "tol"
               "type" : -
	       "default": 1E-6
             },	
             {
               "name" : "quantileProbabilities"
               "type" : list
	       "default": [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
             },
             {
               "name" : "aggregationDepth"
               "type" : int
	       "default": 2
             }
			]
  }
 }
 { linear_params =
  {
   "name" : "LinearRegression"
   "params" : [{
                "name" : "maxIter"
                "type" : int
	        "default": 5
              },
              {
                "name" : "regParam"
                "type" : float
	        "default": 0.01
              },  
              {
                "name" : "elasticNetParam"
                "type" : float
	        "default": 1.0
              },
              {
                "name" : "tol"
                "type" : -
	        "default": 1e-06
              },
              {
                "name" : "fitintercept"
                "type" : bool
	        "default": True
              },
              {
                "name" : "standardization"
                "type" : bool
	        "default": True
              },
              {
                "name" : "solver"
                "type" : -
	        "default": "auto"
              }, 
              {
                "name" : "weightCol"
                "type" : str
	        "default": "weight"
              }, 
              {
                "name" : "aggregationDepth"
                "type" : int
	        "default": 2
              },
              {
                "name" : "loss"
                "type" : -
	        "default": "squaredError"
              }, 
              {
                "name" : "epsilon"
                "type" : float
	        "default": 1.35
              }
			 ]
  }
 }
 { isotonic_params =
  {
   "name" : "IsotonicRegression"
   "params" : [{
                "name" : "predictionCol"
                "type" : str
	        "default": "prediction"
              },
              {
                "name" : "labelCol"
                "type" : str
	        "default": "label"
              },  
              {
                "name" : "featuresCol"
                "type" : str
	        "default": "features"
              },
              {
                "name" : "weightCol"
                "type" : str
	        "default": "weight"
              },
              {
                "name" : "isotonic"
                "type" : bool
	        "default": True
              },
              {
                "name" : "featureIndex"
                "type" : int
	        "default": 0
              }
	     ]
  }
 }
 { dt_params =
  { 
   "name" : "DecisiontreeRegression"
   "params" : [{
                 "name" : "maxDepth"
                 "type" : int
	         "default": 3
               },
               {
                 "name" : "featuresCol"
                 "type" : str
	         "default": "features"
               },  
               {
                 "name" : "labelCol"
                 "type" : str
	         "default": "label"
               },
               {
                 "name" : "predictionCol"
                 "type" : str
	         "default": "prediction"
               },
               {
                 "name" : "maxBins"
                 "type" : int
	         "default": 32
               },
               {
                 "name" : "minInstancesPerNode"
                 "type" : int
	         "default": 1
               },
               {
                 "name" : "minInfoGain"
                 "type" : float
	         "default": 0.0
               }, 
               {
                 "name" : "maxMemoryInMB"
                 "type" : int
	         "default": 256
               }, 
               {
                 "name" : "cacheModeIds"
                 "type" : bool
	         "default": False
               }, 
               {
                 "name" : "checkpointinterval"
                 "type" : int
	         "default": 10
               },
               {
                 "name" : "impurity"
                 "type" : -
	         "default": "variance"
               }, 
               {
                 "name" : "seed"
                 "type" : int
	         "default": None
               },
               {
                 "name" : "varianceCol"
                 "type" : str
	         "default": None
               }
	      ]
  }
 }
 { rfr_params =
  {
   "name" : "RandomforestRegression"
   "params" : [{
                 "name" : "featuresCol"
                 "type" : str
	         "default": "features"
               },  
               {
                 "name" : "labelCol"
                 "type" : str
	         "default": "label"
               },
               {
                 "name" : "predictionCol"
                 "type" : str
	         "default": "prediction"
               },
               {
                 "name" : "maxDepth"
                 "type" : int
	         "default": 5
               },
               {
                 "name" : "maxBins"
                 "type" : int
	         "default": 32
               },
               {
                 "name" : "minInstancesPerNode"
                 "type" : int
	         "default": 1
               },
               {
                 "name" : "minInfoGain"
                 "type" : float
	         "default": 0.0
               }, 
               {
                 "name" : "maxMemoryInMB"
                 "type" : int
	         "default": 256
               }, 
               {
                 "name" : "cacheNodeIds"
                 "type" : bool
	         "default": False
               }, 
               {
                 "name" : "checkpointinterval"
                 "type" : int
	         "default": 10
               },
               {
                 "name" : "impurity"
                 "type" : -
	             "default": "variance"
               }, 
               {
                 "name" : "subsamplingRate"
                 "type" : float
	         "default": 1.0
               },
               {
                 "name" : "seed"
                 "type" : int
	         "default": None
               },
               {
                 "name" : "numTrees"
                 "type" : int
	         "default": 20
               },
               {
                 "name" : "featureSubsetStrategy"
                 "type" : -
	         "default": "auto"
               }
	      ]
  }
 }
 { gbt_params =
  {
   "name" : GradientboostedtreesRegression" 
   "params" :[{
                "name" : "maxIter"
                "type" : int
	        "default": 20
              },  
              {
                "name" : "maxDepth"
                "type" : int
	        "default": 3
              },
              {
                "name" : "featuresCol"
                "type" : str
	        "default": "features"
              },
              {
                "name" : "labelCol"
                "type" : str
	        "default": "label"
              },
              {
                "name" : "predictionCol"
                "type" : str
	        "default": "prediction"
              },
              {
                "name" : "maxBins"
                "type" : int
	        "default": 32
              },
              {
                "name" : "minInstancesPerNode"
                "type" : int
	        "default": 1
              },
              {
                "name" : "minInfoGain"
                "type" : float
	        "default": 0.0
              }, 
              {
                "name" : "maxMemoryInMB"
                "type" : int
	        "default": 256
              }, 
              {
                "name" : "cacheNodeIds"
                "type" : bool
	        "default": False
              }, 
              {
                "name" : "subsamplingRate"
                "type" : float
	        "default": 1.0
              },
              {
                "name" : "checkpointinterval"
                "type" : int
	        "default": 10
              },
              {
                "name" : "lossType"
                "type" : -
	        "default": "squared"
              },
              {
                "name" : "stepSize"
                "type" : float
	        "default": 0.1
              },
              {
                "name" : "seed"
                "type" : int
	        "default": None
              },
              {
                "name" : "impurity"
                "type" : -
	        "default": "variance"
              } 
	     ]
  }
 }
]
