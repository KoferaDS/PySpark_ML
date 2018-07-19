[ 
# Regression Params

 { afts_params = 
  {
   "name" : "aftsurvivalRegression"
   "params" :[{
               "name" : "predictionCol"
               "type" : "str"
	       "default": "prediction"
	       "options" : "any string"
	       "notes" : "prediction column name"
             },
             {
               "name" : "labelCol"
               "type" : "str"
	       "default": "label"
	       "options" : "any string"
	       "notes" : "label column name"
             },
             { 
               "name" : "featuresCol"
               "type" : "str"
	       "default": "features"
	       "options" : "any string"
	       "notes" : "features column name"
             },
             { 
               "name" : "censorCol"
               "type" : "str"
	       "default": "censor"
	       "options" : 0 || 1
	       "notes" : "censor column name. The value of this column could be 0 or 1. If the value is 1, it means the event has occurred i.e. uncensored; otherwise censored"
             },
             { 
               "name" : "quantilesCol"
               "type" : "str"
	       "default": None
	       "options" : "any string"
	       "notes" :"uantiles column name. This column will output quantiles of corresponding quantileProbabilities if it is set"
             },
             { 
               "name" : "fitIntercept"
               "type" : "bool"
	       "default": True
	       "options" : [True||False]
	       "notes" : "whether to fit an intercept term."
             },
             { 
               "name" : "maxIter"
               "type" : "int"
	       "default": 100
	       "options" : >=0
	       "notes" : "max number of iterations"
             },
             { 
               "name" : "tol"
               "type" : "float"
	       "default": 1e-06
	       "options" : >=0
	       "notes" : "the convergence tolerance for iterative algorithms (>= 0)"
             },	
             {
               "name" : "quantileProbabilities"
               "type" : "list"
	       "default": [0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99]
               "options" : range(0,1)
	       "notes" : "Quantile probabilities array. Values of the quantile probabilities array should be in the range (0, 1) and the array should be non-empty"
			 },
             {
               "name" : "aggregationDepth"
               "type" : "int"
	       "default": 2
	       "options" : >=2
	       "notes" : "suggested depth for treeAggregate"
             }
			]
  }
 }
 { linear_params =
  {
   "name" : "linearRegression"
   "params" : [{
                "name" : "maxIter"
                "type" : "int"
	        "default": 100
		"options" : >=0
		"notes" : "max number of iterations (>= 0)"
              },
              {
                "name" : "regParam"
                "type" : "float"
	        "default": 0.01
		"options" : [0,infinite]
		"notes" : "regularization parameter (>= 0)"
              },  
              {
                "name" : "elasticNetParam"
                "type" : "float"
	        "default": 1.0
		"options" : range(0,1)
		"notes" : "the ElasticNet mixing parameter, in range [0, 1]. For alpha = 0, the penalty is an L2 penalty. For alpha = 1, it is an L1 penalty"
              },
              {
                "name" : "tol"
                "type" : "float"
	        "default": 1e-06
		"options" : >=0
		"notes" : "the convergence tolerance for iterative algorithms (>= 0)	"
              },
              {
                "name" : "fitIntercept"
                "type" : "bool"
	        "default": True
		"options" : [True/False]
		"notes" : "whether to fit an intercept term"
              },
              {
                "name" : "standardization"
                "type" : "bool"
	        "default": True
		"options" : [True||False]
		"notes" : "whether to standardize the training features before fitting the model."
              },
              {
                "name" : "solver"
                "type" : "str"
	        "default": "auto"
		"options" : "auto", "normal", "l-bfgs"
		"notes" : "The solver algorithm for optimization."
              }, 
              {
                "name" : "weightCol"
                "type" : "str"
	        "default": None
		"options" : "any string"
		"notes" : "weight column name. If this is not set or empty, we treat all instance weights as 1.0	"
              }, 
              {
                "name" : "aggregationDepth"
                "type" : "int"
	        "default": 2
		"options" : >=2
		"notes" : "suggested depth for treeAggregate"
              },
              {
                "name" : "loss"
                "type" : "str"
	        "default": "squaredError"
		"options" : ["squaredError", "huber"]
		"notes" : "The loss function to be optimized. Supported options: squaredError, huber"
              }, 
              {
                "name" : "epsilon"
                "type" : "float"
	        "default": 1.35
		"options" : >1.0
		"notes" : "The shape parameter to control the amount of robustness. Must be > 1.0. Only valid when loss is huber"
              }
			 ]
  }
 }
 { isotonic_params =
  {
   "name" : "isotonicRegression"
   "params" : [{
                "name" : "predictionCol"
                "type" : "str"
	        "default": "prediction"
		"options" : "any string"
		"notes" : "prediction column name"
              },
              {
                "name" : "labelCol"
                "type" : "str"
	        "default": "label"
		"options" : "any string"
		"notes" : "label column name"
              },  
              {
                "name" : "featuresCol"
                "type" : "str"
	        "default": "features"
		"options" : "any string"
		"notes" : "features column name"
              },
              {
                "name" : "weightCol"
                "type" : "str"
	        "default": "weight"
		"options" : "any string"
		"notes" : "weight column name"
              },
              {
                "name" : "isotonic"
                "type" : "bool"
	        "default": True
		"options" : [True||False]
		"notes" : "whether the output sequence should be isotonic/increasing (true) or" + "antitonic/decreasing (false)"
              },
              {
                "name" : "featureIndex"
                "type" : "int"
	        "default": 0
		"options" : >=0
		"notes" : "The index of the feature if featuresCol is a vector column, no effect otherwise"
              }
	     ]
  }
 }
 { dt_params =
  { 
   "name" : "decisiontreeRegression"
   "params" : [{
                 "name" : "maxDepth"
                 "type" : "int"
	         "default": 3
		 "options" : >=0
		 "notes" : "Maximum depth of the tree. (>= 0) E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes."
               },
               {
                 "name" : "featuresCol"
                 "type" : "str"
	         "default": "features"
		 "options" : "any string"
		 "notes" : "features column name"
               },  
               {
                 "name" : "labelCol"
                 "type" : "str"
	         "default": "label"
		 "options" : "any string"
		 "notes" : "label column name"
               },
               {
                 "name" : "predictionCol"
                 "type" : "str"
	         "default": "prediction"
		 "options" : "any string"
		 "notes" : "prediction column name"
               },
               {
                 "name" : "maxBins"
                 "type" : "int"
	         "default": 32
		 "options" : >=2
		 "notes" : "Max number of bins for discretizing continuous features (Must be >=2 and >= number of categories for any categorical feature)"
               },
               {
                 "name" : "minInstancesPerNode"
                 "type" : "int"
	         "default": 1
		 "options" : >=1
		 "notes" : "Minimum number of instances each child must have after split. If a split causes the left or right child to have fewer than minInstancesPerNode, the split will be discarded as invalid. "
               },
               {
                 "name" : "minInfoGain"
                 "type" : "float"
	         "default": 0.0
		 "options" : [0,infinite]
		 "notes" : "Minimum information gain for a split to be considered at a tree node"
               }, 
               {
                 "name" : "maxMemoryInMB"
                 "type" : "int"
	         "default": 256
		 "options" : [0,infinite]
		 "notes" : "Maximum memory in MB allocated to histogram aggregation. If too small, then 1 node will be split per iteration, and its aggregates may exceed this size"
               }, 
               {
                 "name" : "cacheNodeIds"
                 "type" : "bool"
	         "default": False
		 "options" : [True||False]
		 "notes" : "If false, the algorithm will pass trees to executors to match instances with nodes. If true, the algorithm will cache node IDs for each instance. Caching can speed up training of deeper trees. Users can set how often should the cache be checkpointed or disable it by setting checkpointInterval"
               }, 
               {
                 "name" : "checkpointerval"
                 "type" : "int"
	         "default": 10
		 "options" : "set checkpoint interval (>= 1) or disable checkpoint (-1)."
		 "notes" : " E.g. 10 means that the cache will get checkpointed every 10 iterations. Note: this setting will be ignored if the checkpoint directory is not set in the SparkContext"
               },
               {
                 "name" : "impurity"
                 "type" : "str"
	         "default": "variance"
		 "options" :"variance"
		 "notes" : "Criterion used for information gain calculation (case-insensitive)"
               }, 
               {
                 "name" : "seed"
                 "type" : "int"
	         "default": None
		 "options" : [0,infinite]
		 "notes" : "random seed"
               },
               {
                 "name" : "varianceCol"
                 "type" : "str"
	         "default": None
		 "options" : "any string"
		 "notes" : "column name for the biased sample variance of prediction	"
               }
	      ]
  }
 }
 { rfr_params =
  {
   "name" : "randomforestRegression"
   "params" : [{
                 "name" : "featuresCol"
                 "type" : "str"
	         "default": "features"
		 "options" : "any string"
		 "notes" : "features column name"
               },  
               {
                 "name" : "labelCol"
                 "type" : "str"
	         "default": "label"
		 "options" : "any string"
		 "notes" : "label column name"
               },
               {
                 "name" : "predictionCol"
                 "type" : "str"
	         "default": "prediction"
		 "options" : "any string"
		 "notes" : "prediction column name"
               },
               {
                 "name" : "maxDepth"
                 "type" : "int"
	         "default": 5
		 "options" : >=0
		 "notes" : "Maximum depth of the tree. (>= 0) E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes."
               },
               {
                 "name" : "maxBins"
                 "type" : "int"
	         "default": 32
		 "options" : >=2
		 "notes" : "Max number of bins for discretizing continuous features (Must be >=2 and >= number of categories for any categorical feature)"
               },
               {
                 "name" : "minInstancesPerNode"
                 "type" : "int"
	         "default": 1
		 "options" : >=0
		 "notes" : "Minimum number of instances each child must have after split. If a split causes the left or right child to have fewer than minInstancesPerNode, the split will be discarded as invalid. "
               },
               {
                 "name" : "minInfoGain"
                 "type" : "float"
	         "default": 0.0
		 "options" : [0,infinite]
		 "notes" : "Minimum information gain for a split to be considered at a tree node"
               }, 
               {
                 "name" : "maxMemoryInMB"
                 "type" : "int"
	         "default": 256
		 "options" : [0,infinite]
		 "notes" : "Maximum memory in MB allocated to histogram aggregation. If too small, then 1 node will be split per iteration, and its aggregates may exceed this size"
               }, 
               {
                 "name" : "cacheNodeIds"
                 "type" : "bool"
	         "default": False
		 "options" :[True||False]
		 "notes" : "If false, the algorithm will pass trees to executors to match instances with nodes. If true, the algorithm will cache node IDs for each instance. Caching can speed up training of deeper trees. Users can set how often should the cache be checkpointed or disable it by setting checkpointInterval."
               }, 
               {
                 "name" : "checkpoInterval"
                 "type" : "int"
	         "default": 10
		 "options" : "set checkpoint interval (>= 1) or disable checkpoint (-1)."
		 "notes" : "E.g. 10 means that the cache will get checkpointed every 10 iterations. Note: this setting will be ignored if the checkpoint directory is not set in the SparkContext."
               },
               {
                 "name" : "impurity"
                 "type" : "str"
	         "default": "variance"
		 "options" : "variance"
		 "notes" : "Criterion used for information gain calculation (case-insensitive)"
               }, 
               {
                 "name" : "subsamplingRate"
                 "type" : "float"
	         "default": 1.0
		 "options" : range(0,1)
		 "notes" : "Fraction of the training data used for learning each decision tree, in range [0, 1]."
               },
               {
                 "name" : "seed"
                 "type" : "int"
	         "default": None
		 "options" : [0,infinite]
		 "notes" : "random seed"
               },
               {
                 "name" : "numTrees"
                 "type" : "int"
	         "default": 20
		 "options" : >=1
		 "notes" : "Number of trees to train (>= 1)."
               },
               {
                 "name" : "featureSubsetstrategy"
                 "type" : "str"
	         "default": "auto"
		 "options" : ["auto", "all", "onethird", "sqrt", "log2", (0.0-1.0], (1-n)]
		 "notes" : "The number of features to consider for splits at each tree node."
               }
	      ]
  }
 }
 { gbt_params =
  {
   "name" : gradientboostedtreesRegression" 
   "params" :[{
                "name" : "maxIter"
                "type" : "int"
	        "default": 20
		"options" : >=0
		"notes" : "max number of iterations	"
              },  
              {
                "name" : "maxDepth"
                "type" : "int"
	        "default": 5
		"options" : >=0
		"notes" : "Maximum depth of the tree. (>= 0) E.g., depth 0 means 1 leaf node; depth 1 means 1 internal node + 2 leaf nodes."
              },
              {
                "name" : "featuresCol"
                "type" : "str"
	        "default": "features"
		"options" : "any string"
		"notes" : "features column name"
              },
              {
                "name" : "labelCol"
                "type" : "str"
	        "default": "label"
		"options" : "any string"
		"notes" : "label column name"
              },
              {
                "name" : "predictionCol"
                "type" : "str"
	        "default": "prediction"
		"options" : "any string"
		"notes" : "prediction column name"
              },
              {
                "name" : "maxBins"
                "type" : "int"
	        "default": 32
		"options" : >=2
		"notes" : "Max number of bins for discretizing continuous features (Must be >=2 and >= number of categories for any categorical feature)	"
              },
              {
                "name" : "minInstancesPerNode"
                "type" : "int"
	        "default": 1
		"options" : >=1
		"notes" : "Minimum number of instances each child must have after split. If a split causes the left or right child to have fewer than minInstancesPerNode, the split will be discarded as invalid. 	"
              },
              {
                "name" : "minInfoGain"
                "type" : "float"
	        "default": 0.0
		"options" : [0,infinite]
		"notes" : "Minimum information gain for a split to be considered at a tree node	"
              }, 
              {
                "name" : "maxMemoryInMB"
                "type" : "int"
	        "default": 256
		"options" : [0,infinite]
		"notes" : "Maximum memory in MB allocated to histogram aggregation. If too small, then 1 node will be split per iteration, and its aggregates may exceed this size	"
              }, 
              {
                "name" : "cacheNodeIds"
                "type" : "bool"
	        "default": False
		"options" : [True||False]
		"notes" : "If false, the algorithm will pass trees to executors to match instances with nodes. If true, the algorithm will cache node IDs for each instance. Caching can speed up training of deeper trees. Users can set how often should the cache be checkpointed or disable it by setting checkpointInterval"
              }, 
              {
                "name" : "subsamplingRate"
                "type" : "float"
	        "default": 1.0
		"options" : range(0,1)
		"notes" : "Fraction of the training data used for learning each decision tree, in range (0, 1]	"
              },
              {
                "name" : "checkpoInterval"
                "type" : "int"
	        "default": 10
		"options" : "set checkpoint interval (>= 1) or disable checkpoint (-1)"
		"notes" : " E.g. 10 means that the cache will get checkpointed every 10 iterations. Note: this setting will be ignored if the checkpoint directory is not set in the SparkContext	"
              },
              {
                "name" : "lossType"
                "type" : "str"
	        "default": "squared"
		"options" : ["squared", "huber"]
		"notes" : "The loss function to be optimized. 	"
              },
              {
                "name" : "stepSize"
                "type" : "float"
	        "default": 0.1
		"options" : range(0,1)
		"notes" : "Step size (a.k.a. learning rate) in interval (0, 1] for shrinking the contribution of each estimator.	"
              },
              {
                "name" : "seed"
                "type" : "int"
	        "default": None
		"options" : [0,infinite]
		"notes" : "random seed"
              },
              {
                "name" : "impurity"
                "type" : "str"
	        "default": "variance"
		"options" : "variance"
		"notes" : "Criterion used for information gain calculation (case-insensitive)"
              } 
	     ]
  }
 }
]
