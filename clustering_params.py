#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 13:45:09 2018

@author: Mujirin, mujirin@kofera.com
"""

bisecting_kmeans_params = {
     "name": "bisectingKmeans",
     "params":[
              {
                "name" : "featuresCol",
                "type" : "str",
                "default": "features",
                "value": None,
                "options" : "any string",
                "notes" : "features column name"
              },
              {
                "name" : "k",
                "type" : "int",
                "default": 4,
                "value": None,
                "options" : '> 1',
                "notes" : "the desired number of leaf cluster"
              },  
              {
                "name" : "maxIter",
                "type" : "int",
                "default": 20,
                "value": None,
                "options" : ">=0",
                "notes" : "maximum number of iterations"
              },
              {
                "name" : "minDivisibleClusterSize",
                "type" : "float",
                "default": 1.0,
                "value": None,
                "options" : 'any float number',
                "notes" : "the minimum number of points (if >= 1.0) or the minimum proportion of points (if < 1.0) of a divisible cluster"
              },
              {
                "name" : "predictionCol",
                "type" : "str",
                "default": 'prediction',
                "value": None,
                "options" : "any string",
                "notes" : "prediction column name"
              },
              {
                "name" : "seed",
                "type" : "int",
                "default": None,
                "value": None,
                "options" : "any integer",
                "notes" : "Random seed, must be Integer, float number will error exept 0 after comma"
              }
	          ]
             }
     
gaussian_mixture_params = {
     "name": "gaussianMixture",
     "params":[
              {
                "name" : "featuresCol",
                "type" : "str",
                "default": "features",
                "value": None,
                "options" : "any string",
                "notes" : "features column name"
              },
              {
                "name" : "k",
                "type" : "int",
                "default": 2,
                "value": None,
                "options" : '>1',
                "notes" : "the desired number of leaf cluster"
              },  
              {
                "name" : "maxIter",
                "type" : "int",
                "default": 100,
                "value": None,
                "options" : ">=0",
                "notes" : "maximum number of iterations"
              },
              {
                "name" : "predictionCol",
                "type" : "str",
                "default": 'prediction',
                "value": None,
                "options" : "any string",
                "notes" : "prediction column name"
              },
              {
                "name" : "probabilityCol",
                "type" : "str",
                "default": "probability",
                "value": None,
                "options" : 'any string',
                "notes" : "cColumn name for predicted class conditional probabilities. Note: Not all models output well-calibrated probability estimates! These probabilities should be treated as confidences, not precise probabilities'"
              },
              {
                "name" : "seed",
                "type" : "int",
                "default": None,
                "value": None,
                "options" : "any integer",
                "notes" : "Random seed, must be Integer, float number will error exept 0 after comma"
              },
              {
                "name" : "tol",
                "type" : "float",
                "default": 0.01,
                "value": None,
                "options" : ">=0",
                "notes" : "the convergence tolerance for iterative algorithms"
              }
	          ]
             }
     
kmeans_params = {
     "name": "kmeans",
     "params":[
              {
                "name" : "featuresCol",
                "type" : "str",
                "default": "features",
                "value": None,
                "options" : "any string",
                "notes" : "features column name"
              },
              {
                "name" : "initMode",
                "type" : "str",
                "default": 'k-means||',
                "value": None,
                "options" : "'random' or 'k-means||'",
                "notes" : "the initialization algorithm. Supported options: 'random' and 'k-means||'"
              },
              {
                "name" : "initSteps",
                "type" : "int",
                "default": 2,
                "value": None,
                "options" : ">0",
                "notes" : "the number of steps for k-means|| initialization mode. Must be > 0"
              },
              {
                "name" : "k",
                "type" : "int",
                "default": 2,
                "value": None,
                "options" : '>1',
                "notes" : "the number of clusters to create. Must be > 1"
              }, 
              {
                "name" : "maxIter",
                "type" : "int",
                "default": 20,
                "value": None,
                "options" : ">=0",
                "notes" : "maximum number of iterations"
              },
              {
                "name" : "predictionCol",
                "type" : "str",
                "default": 'prediction',
                "value": None,
                "options" : "any string",
                "notes" : "prediction column name"
              },
              {
                "name" : "seed",
                "type" : "int",
                "default": None,
                "value": None,
                "options" : "any integer",
                "notes" : "random seed, must be Integer, float number will error exept 0 after comma"
              },     
              {
                "name" : "tol",
                "type" : "float",
                "default": 0.0001,
                "value": None,
                "options" : ">=0",
                "notes" : "the convergence tolerance for iterative algorithms"
              }
	          ]
             }
     
lda_params = {
     "name": "lda",
     "params":[
              {
                "name" : "checkpointInterval",
                "type" : "str",
                "default": 10,
                "value": None,
                "options" : ">=1 set check point, (-1) for disable. ",
                "notes" : "Set checkpoint interval (>= 1) or disable checkpoint (-1). E.g. 10 means that the cache will get checkpointed every 10 iterations, this setting will be ignored if the checkpoint directory is not set in the SparkContext"
              },
              {
                "name" : "featuresCol",
                "type" : "str",
                "default": "features",
                "value": None,
                "options" : "any string",
                "notes" : "features column name"
              },
              {
                "name" : "k",
                "type" : "int",
                "default": 10,
                "value": None,
                "options" : '>1',
                "notes" : "the number of topics (clusters) to infer"
              },
              {
                "name" : "keepLastCheckpoint",
                "type" : "boolean",
                "default": 'true',
                "value": None,
                "options" : "true or false",
                "notes" : "(for em optimizer) if using checkpointing, this indicates whether to keep the last checkpoint. if false, then the checkpoint will be deleted. deleting the checkpoint can cause failures if a data partition is lost, so set this bit with care"
              },
              {
                "name" : "learningDecay",
                "type" : "float",
                "default": 0.51,
                "value": None,
                "options" : "(0.5, 1.0]",
                "notes" : "this should be between (0.5, 1.0] to guarantee asymptotic convergence. (for online optimizer) learning rate, set as an exponential decay rate"
              },       
              {
                "name" : "learningOffset",
                "type" : "float",
                "default": 1024.0,
                "value": None,
                "options" : ">0.0",
                "notes" : "(for online optimizer) a (positive) learning parameter that downweights early iterations. larger values make early iterations count less"
              },        
              {
                "name" : "maxIter",
                "type" : "int",
                "default": 20,
                "value": None,
                "options" : ">=0",
                "notes" : "maximum number of iterations"
              },       
              {
                "name" : "optimizeDocConcentration",
                "type" : "boolean",
                "default": 'true',
                "value": None,
                "options" : "true or false",
                "notes" : "(For online optimizer only, currently) Indicates whether the docConcentration (Dirichlet parameter for document-topic distribution) will be optimized during training"
              },
              {
                "name" : "optimizer",
                "type" : "str",
                "default": 'online',
                "value": None,
                "options" : "'online' or 'em'",
                "notes" : "optimizer or inference algorithm used to estimate the LDA model"
              },
              {
                "name" : "seed",
                "type" : "int",
                "default": None,
                "value": None,
                "options" : "any integer",
                "notes" : "Random seed, must be Integer, float number will error exept 0 after comma"
              },
              {
                "name" : "subsamplingRate",
                "type" : "float",
                "default": 0.05,
                "value": None,
                "options" : "(0,1]",
                "notes" : "(for online optimizer) fraction of the corpus to be sampled and used in each iteration of mini-batch gradient descent"
              },          
              {
                "name" : "topicDistributionCol",
                "type" : "str",
                "default": 'topicDistribution',
                "value": None,
                "options" : "any string",
                "notes" : "output column with estimates of the topic mixture distribution for each document (often called 'theta' in the literature). returns a vector of zeros for an empty document"
              },        
              {
                "name" : "topicConcentration",
                "type" : "float",
                "default": None,
                "value": None,
                "options" : ">1.0 or -1 for auto",
                "notes" : "concentration parameter (commonly named 'beta' or 'eta') for the prior placed on topic\' distributions over terms"
              }]}