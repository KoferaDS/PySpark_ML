[ 
# Pre-processing Params
 { minmax_params = 
  {
  "name" : "MinMaxScaler"
  "params" :[{
               "name" : "min"
               "type" : float
	       "default": 0.0  
             },
             {
               "name" : "max"
               "type" : float
	       "default": 1.0
             },
             { 
               "name" : "inputCol"
               "type" : str
	       "default": "features"
             },
             { 
               "name" : "outputCol"
               "type" : str
	       "default": "scaledFeatures"
             }
			]
  }
 }
 { maxabs_params =
  {
  "name" : "MaxAbsScaler"
  "params" : [{
                "name" : "inputCol"
                "type" : str
	        "default": "features"
              },
              {
                "name" : "outputCol"
                "type" : str
	        "default": "scaledFeatures"
              } 
 	     ]
  }
 }
 { normalizer_params =
  {
  "name" : "Normalizer"
  "params" : [{
                "name" : "p"
                "type" : float
	        "default": 2.0
              },
              {
                "name" : "inputCol"
                "type" : str
	        "default": None
              },
              {
                "name" : "outputCol"
                "type" : str
	        "default": None
              } 
 	     ]
  }
 }
 { standardscaler_params =
  {
  "name" : "StandardScaler"
  "params" : [{
                "name" : "withMean"
                "type" : bool
	        "default": False
              },
              {
                "name" : "withStd"
                "type" : bool
	        "default": True
              },  
              {
                "name" : "inputCol"
                "type" : str
	        "default": "features"
              },
              {
                "name" : "outputCol"
                "type" : str
	        "default": "scaledFeatures"
              }
	     ]
  }
 }
 { indextostring_params =
  {
  "name" : "IndexToString"
  "params" : [{
                "name" : "inputCol"
                "type" : str
	            "default": None
              },
              {
                "name" : "outputCol"
                "type" : str
	            "default": None
              }
			                {
                "name" : "labels"
                "type" : str
	            "default": None
              }
	     ]
  }
 }
 { stringindexer_params =
  {
  "name" : "StringIndexer"
  "params" : [{
                "name" : "inputCol"
                "type" : str
	        "default": None
              },
              {
                "name" : "outputCol"
                "type" : str
	        "default": None
              }
			                {
                "name" : "handledInvalid
                "type" : str
	        "default": "error"
              }
	     ]
  }
 } 
 { PCA_params =
  { 
   "name" : "PCA"
   "params" : [{
                 "name" : "k"
                 "type" : int
	         "default": None
               },
               {
                 "name" : "inputCol"
                 "type" : str
	         "default": None
               },  
               {
                 "name" : "outputCol"
                 "type" : str
	         "default": None
               }
	      ]
  }
 }
]
