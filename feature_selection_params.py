#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 25 13:45:09 2018

@author: Mujirin, mujirin@kofera.com
"""

dct_params = {
     "name": "dct",
     "params":[
              {
                "name" : "outputCol",
                "type" : "str",
                "default": None,
                "value": None,
                "options" : "any string",
                "notes" : "output column name"
              },
              {
                "name" : "inputCol",
                "type" : "str",
                "default": None,
                "value": None,
                "options" : "any string",
                "notes" : "input column name"
              },  
              {
                "name" : "inverse",
                "type" : "boolean",
                "default": "false",
                "value": None,
                "options" : "'true' or 'false'",
                "notes" : "set transformer to perform inverse dct, default false"
              }
	          ]
             }
     
pca_params = {
     "name": "pca",
     "params":[
              {
                "name" : "outputCol",
                "type" : "str",
                "default": None,
                "value": None,
                "options" : "any string",
                "notes" : "output column name"
              },
              {
                "name" : "inputCol",
                "type" : "str",
                "default": None,
                "value": None,
                "options" : "any string",
                "notes" : "input column name"
              },  
              {
                "name" : "k",
                "type" : "int",
                "default": None,
                "value": None,
                "options" : ">0",
                "notes" : "the number of principal components "
              }
	          ]
             }