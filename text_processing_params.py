[

	# Text processing params

	{ n_gram_params = {
		"name" : "nGram",
		"params" : [
					{
						"name" : "n",
						"type" : "int",
						"default" : 2,
						"options" : "n must great or equal than 1",
						"notes" : ""
					},
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
					}
					]

		}
	}

	{ hashing_tf_params = {
		"name" : "hashingTF",
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
						"name" : "binary",
						"type" : "bool",
						"default" : False,
						"options" : "",
						"notes" : ""
					},
					{
						"name" : "numFeatures",
						"type" : "int",
						"default" : 262144,
						"options" : "",
						"notes" : ""
					}
					]

		}
	}

	{ count_vectorizer_params = {
		"name" : "countVectorizer",
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
						"type" :, "string"
						"default" : None,
						"options" : "",
						"notes" : ""
					},
					{
						"name" : "minTF",
						"type" : "double", 
						"default" : 1.0,
						"options" : "",
						"notes" : ""
					},
					{
						"name" : "minDF",
						"type" : "double",
						"default" : 1.0,
						"options" : "",
						"notes" : ""
					},
					{
						"name" : "vocabSize",
						"type" : "int",
						"default" : 262144,
						"options" : "",
						"notes" : ""
					},
					{
						"name" : "binary",
						"type" : "bool",
						"default" : False,
						"options" : "",
						"notes" : ""
					},
					]

		}
	}

	{ idf_params = {
		"name" : "IDF",
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
						"name" : "minDocFreq",
						"type" : "int",
						"default" : 0,
						"options" : "must positive",
						"notes" : ""
					}
					]

		}
	}

	{ word2vec_params = {
		"name" : "word2Vec",
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
						"name" : "minCount",
						"type" : "int",
						"default" : 5,
						"options" : "must positive",
						"notes" : ""
					},
					{
						"name" : "numPartitions",
						"type" : "int",
						"default" : 1,
						"options" : "must positive",
						"notes" : ""
					},
					{
						"name" : "stepSize",
						"type" : "float",
						"default" : 0.025,
						"options" : "must positive",
						"notes" : ""
					},
					{
						"name" : "maxIter",
						"type" : "int",
						"default" : 1,
						"options" : "must positive",
						"notes" : ""
					},
					{
						"name" : "seed",
						"type" : "double",
						"default" : None,
						"options" : "",
						"notes" : ""
					},
					{
						"name" : "maxSentenceLength",
						"type" : "int", 
						"default" : 1000,
						"options" : "must positive",
						"notes" : ""
					},
					{
						"name" : "windowSize",
						"type" : "int",
						"default" : 5,
						"options" : "must positive",
						"notes" : ""
					}
					{
						"name" : "vectorSize",
						"type" : "int",
						"default" : 100,
						"options" : "must positive",
						"notes" : ""
					}
					]

		}
	}


]