[ 
# Pre-processing Params
{ minmax_params = {
		"name" : "MinMaxScaler",
		"params" : [
			{
				"name" : "min",
				"type" : "float",
				"default" : 0.0,
				"options" : "any",
				"notes" : ""
			},
			{
				"name" : "max",
				"type" : "float",
				"default": 1.0,
				"options" : "any",
				"notes" : ""
			},
			{ 
				"name" : "inputCol",
				"type" : "string",
				"default": "features",
				"options" : "any",
				"notes" : ""
			},
			{ 
				"name" : "outputCol",
				"type" : "string",
				"default": "scaledFeatures",
				"options" : "any",
				"notes" : ""
			}
			]
  		}
 	}
 { maxabs_params = {
		"name" : "MaxAbsScaler",
		"params" : [
			{ 
				"name" : "inputCol",
				"type" : "string",
				"default": "features",
				"options" : "any",
				"notes" : ""
			},
			{ 
				"name" : "outputCol",
				"type" : "string",
				"default": "scaledFeatures",
				"options" : "any",
				"notes" : ""
			}
			]
  		}
 	}
 { binarizer_params = {
		"name" : "BinaryNormalization",
		"params" : [
			{
				"name" : "threshold",
				"type" : "float",
				"default" : 0.0,
				"options" : "range [0.1]",
				"notes" : ""
			},
			{ 
				"name" : "inputCol",
				"type" : "string",
				"default": "features",
				"options" : "any",
				"notes" : ""
			},
			{ 
				"name" : "outputCol",
				"type" : "string",
				"default": "scaledFeatures",
				"options" : "any",
				"notes" : ""
			}
			]
  		}
 	}
 { standardscaler_params ={
		"name" : "StandardScaler",
		"params" : [
			{
				"name" : "withMean",
				"type" : "boolean",
				"default" : False,
				"options" : "any",
				"notes" : ""
			},
			{
				"name" : "withStd",
				"type" : "boolean",
				"default": True,
				"options" : "any",
				"notes" : ""
			},
			{ 
				"name" : "inputCol",
				"type" : "string",
				"default": "features",
				"options" : "any",
				"notes" : ""
			},
			{ 
				"name" : "outputCol",
				"type" : "string",
				"default": "scaledFeatures",
				"options" : "any",
				"notes" : ""
			}
			]
  		}
 	}
#  { indextostring_params =
#   {
#   "name" : "IndexToString"
#   "params" : [{
#                 "name" : "inputCol"
#                 "type" : str
# 	            "default": None
#               },
#               {
#                 "name" : "outputCol"
#                 "type" : str
# 	            "default": None
#               }
# 			                {
#                 "name" : "labels"
#                 "type" : str
# 	            "default": None
#               }
# 	     ]
#   }
#  }
#  { stringindexer_params =
#   {
#   "name" : "StringIndexer"
#   "params" : [{
#                 "name" : "inputCol"
#                 "type" : str
# 	        "default": None
#               },
#               {
#                 "name" : "outputCol"
#                 "type" : str
# 	        "default": None
#               }
# 			                {
#                 "name" : "handledInvalid"
#                 "type" : str
# 	        "default": "error"
#               }
# 	     ]
#   }
#  } 
#  { PCA_params =
#   { 
#    "name" : "PCA"
#    "params" : [{
#                  "name" : "k"
#                  "type" : int
# 	         "default": None
#                },
#                {
#                  "name" : "inputCol"
#                  "type" : str
# 	         "default": None
#                },  
#                {
#                  "name" : "outputCol"
#                  "type" : str
# 	         "default": None
#                }
# 	      ]
#   }
#  }
]
