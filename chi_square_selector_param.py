[

	# Chi Square Selector Params

	{ chi_square_selector_params = {
		"name" : "chiSquareSelector",
		"params" : [
					{
						"name" : "inputCol",
						"type" : "string",
						"default" : None,
						"options" : "",
						"notes" : ""
					},
					{
						"name" : "outputCol",
						"type" : "string",
						"default" : None,
						"options" : "",
						"notes" : ""
					},
					{
						"name" : "featuresCol",
						"type" : "string",
						"default" : "features",
						"options" : "",
						"notes" : ""
					},
					{
						"name" : "selectorType",
						"type" : "string",
						"default" : "numTopFeatures",
						"options" : "",
						"notes" : {
									"selectorType" : ["numTopFeatures", "percentile", "fpr"]
									}	 
					},
					{
						"name" : "numTopFeatures",
						"type" : "int",
						"default" : 50,
						"options" : "",
						"notes" : ""
					},
					{
						"name" : "percentile",
						"type" : "float",
						"default" : 0.1,
						"options" : "between 0 and 1",
						"notes" : ""
					},
					{
						"name" : "fpr",
						"type" : "float",
						"default" : 0.05,
						"options" : "between 0 and 1",
						"notes" : ""
					},
					]
		}
	}
]