#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 11 01:43:41 2018

@author: Mujirin
email: mujirin@kofera.com
"""
# =============================================================================
# Latent Dirichlet Allocation (LDA), a topic model designed for text documents.
# =============================================================================
from pyspark.ml.linalg import Vectors, SparseVector
from pyspark.ml.clustering import LDA
#from pyspark.ml.clustering.
from spark_sklearn.util import createLocalSparkSession
spark = createLocalSparkSession()

def hiperAdapter(hiperparameter):
    '''
    Fungsi ini untuk menyesuaikan config 
    yang tidak lengkap
    ke defaultnya.
    Input: User hiperparameter setting
    Output: library of hiperparameter with the default
    Params:
            k: Number of topics (i.e.,)
            optimizer:  'em'        ---> topicConcentration must be > 1.0
                                    ---> docConcentration must be > 1.0 
                        'online'
            docConcentration: Dirichlet parameter for prior over documents’ 
                                distributions over topics. 
                                Larger values encourage smoother inferred distributions.
                                either 'em' or 'online' the devalult cannot be None.
            topicConcentration: Dirichlet parameter for prior over topics’ 
                                distributions over terms (words). 
                                Larger values encourage smoother inferred distributions.
            maxIterations: Limit on the number of iterations.
            checkpointInterval: If using checkpointing (set in the Spark configuration), 
                                this parameter specifies the frequency 
                                with which checkpoints will be created. 
                                If maxIterations is large, using checkpointing can 
                                help reduce shuffle file sizes on disk and help with 
                                failure recovery.
    Warning:
    In this module we do not include docConcentration = None, because its error
    in either 'em' or 'online' optimizer.        
    '''
    hiperparameter_default = {"featuresCol":"features", 
                              "maxIter":20,
                              "seed":None,
                              "checkpointInterval":10,
                              "k":10,
                              "optimizer":"online",
                              "learningOffset":1024.0,
                              "learningDecay":0.51,
                              "subsamplingRate":0.05,
                              "optimizeDocConcentration":True,
                              #"docConcentration":None,
                              "topicConcentration":None,
                              "topicDistributionCol":"topicDistribution",
                              "keepLastCheckpoint":True}
    
    hiperparameter_keys = list(hiperparameter.keys())
    hiperparameter_def_keys = list(hiperparameter_default.keys())
    new_hiperparameter = {}
    for hip in hiperparameter_def_keys:
        if hip not in hiperparameter_keys:
            new_hiperparameter[hip] = hiperparameter_default[hip]
        else:
            new_hiperparameter[hip] = hiperparameter[hip]
    return new_hiperparameter

def train(df,hiperparameter):
    '''
    LDA training, returning LDA model.
    input: - Dataframe
           - config (configurasi hiperparameter)
    
    return: kmeans model
    '''
    lda = LDA(featuresCol = hiperparameter['featuresCol'],
              maxIter = hiperparameter['maxIter'],
              seed = hiperparameter['seed'],
              checkpointInterval = hiperparameter['checkpointInterval'],
              k = hiperparameter['k'],
              optimizer = hiperparameter['optimizer'],
              learningOffset = hiperparameter['learningOffset'],
              learningDecay = hiperparameter['learningDecay'],
              subsamplingRate = hiperparameter['subsamplingRate'],
              optimizeDocConcentration = hiperparameter['optimizeDocConcentration'],
#              docConcentration = hiperparameter['docConcentration'],
              topicConcentration = hiperparameter['topicConcentration'],
              topicDistributionCol = hiperparameter['topicDistributionCol'],
              keepLastCheckpoint = hiperparameter['keepLastCheckpoint'])
    
    model = lda.fit(df)
    return model

def copyModel(model):
    '''
    Returning copied model
    Input: model
    Output: model of copy
    '''
    copy_model = model.copy(extra=None)
    return copy_model

def saveModel(model,path):
    '''
    Save model into corresponding path.
    Input:  - model
            - path
    Output: saved model
    '''
    model.save(path)
    return

def loadModel(model,path):
    '''
    Loading model from path.
    Input: model
    Output: loaded model
    ***catatan:
        ini kasus yang sangat berbeda, load model membutuhkan model
        mungkin karena yang buat di pyspark lupa kalau load harusnya di LDAModel tempatnya, atau ada faktor lain
    '''
    model = model.load(path)
    return model

def topicsDescription(model):
    '''
    Return the topics described by their top-weighted terms.
    Input: model
    output: dataframe with topic, termIndices and termWeights columns
    '''
    topics_desc = model.describeTopics()
    
    return topics_desc

def docConcentrationEstimation(model):
    '''
    Value for LDA.docConcentration estimated from data. If Online LDA was used and LDA.optimizeDocConcentration was set to false, then this returns the fixed (given) value for the LDA.docConcentration parameter.(Pyspark documentation)
    Input: model
    Output: 
    '''
    doc_concentration = [(model.estimatedDocConcentration(),)]
    df_doc_concentration = spark.createDataFrame(doc_concentration, ["estimatedDocConcentration"])
    return df_doc_concentration

def modelLikelihood(model,df):
    '''
    Calculates a lower bound on the log likelihood of the entire corpus. See Equation (16) in the Online LDA paper (Hoffman et al., 2010).
    WARNING: If this model is an instance of DistributedLDAModel 
    (produced when optimizer is set to “em”), 
    this involves collecting a large topicsMatrix() to the driver. 
    This implementation may be changed in the future.(Pyspark documentation)
    Returning likelihood of dataframe and model.
    Input:  - model
            - dataframe
    '''
    likelihood = model.logLikelihood(df)
    likelihood = [(Vectors.dense(likelihood),)]
    df_likelihood = spark.createDataFrame(likelihood, ["Likelihood"])
    return df_likelihood

def modelPerplexity(model,df):
    '''
    Calculate an upper bound on perplexity. 
    (Lower is better.) See Equation (16) 
    in the Online LDA paper (Hoffman et al., 2010).
    WARNING: If this model is an instance of DistributedLDAModel 
    (produced when optimizer is set to “em”), 
    this involves collecting a large topicsMatrix() to the driver. 
    This implementation may be changed in the future.(Pyspark documentation)
    Returning dataframe of perplexity of the model.
    Input:  - model
            - dataframe
    '''
    perplexity = model.logPerplexity(df)
    perplexity = [(Vectors.dense(perplexity),)]
    df_perplexity = spark.createDataFrame(perplexity, ["Perplexity"])
    return df_perplexity

def logPriorModel(model):
    '''
    Log probability of the current parameter estimate: 
    log P(topics, topic distributions for docs | alpha, eta)
    Input: model
    Output: dataframe of log prior of the model
    **catatan: Untuk sementara hanya bisa cek logPrior untuk defalult
    '''
    prior = model.logPrior()
    prior = [(Vectors.dense(prior),)]
    df_prior = spark.createDataFrame(prior, ["LogPrior"])
    return df_prior

def topicsMatrixModel(model):
    '''
    Inferred topics, where each topic is represented by a 
    distribution over terms. This is a matrix of size vocabSize x k, 
    where each column is a topic. No guarantees 
    are given about the ordering of the topics.
    WARNING: If this model is actually
    a DistributedLDAModel instance produced 
    by the Expectation-Maximization (“em”) optimizer,
    then this method could involve collecting 
    a large amount of data to the driver 
    (on the order of vocabSize x k).
    Input: model
    Ouput: Dataframe of topicsMatrix
    '''
    topic_mat = model.topicsMatrix()
    topic_mat = [(topic_mat,)]
    df_topic_mat = spark.createDataFrame(topic_mat, ["Topics Matrix"])
    return df_topic_mat

def trainingLogLikelihoodModel(model):
    '''
    Log likelihood of the observed tokens in the training set, 
    given the current parameter estimates: log P(docs | topics, 
    topic distributions for docs, Dirichlet hyperparameters)
    Notes:
    This excludes the prior; for that, use logPrior().
    Even with logPrior(), this is NOT the same as the data 
    log likelihood given the hyperparameters.
    This is computed from the topic distributions computed
    during training. If you call logLikelihood() 
    on the same training dataset,
    the topic distributions will be computed again, 
    possibly giving different results(Pyspark documentation).
    Input: model
    output: dataframe of training log likelihood of the model
    '''
    training_log = model.trainingLogLikelihood()
    training_log = [(Vectors.dense(training_log),)]
    df_training_log = spark.createDataFrame(training_log, ["Training log likelihood"])
    return df_training_log
    
def predictDF(model, df, params=None):
    '''
    Transforms the input dataset with optional parameters.
    Parameters:	
        dataset – input dataset, which is an instance of pyspark.sql.DataFrame
        params – an optional param map that overrides embedded params.
    Input: model
           df
    Output: transformed dataframe with tipic distribution as an one of the its coloumn
    '''
    pred_df = model.transform(df, params)
    return pred_df

def vocabSizeModel(model):
    '''
    Vocabulary size (number of terms or words in the vocabulary)
    Input: model
    Output: vocab size dataframe
    '''
    voc_size = model.vocabSize()
    voc_size = [(Vectors.dense(voc_size),)]
    df_voc_size = spark.createDataFrame(voc_size, ["Vocab size"])
    return df_voc_size

def paramsExplanation(model):
    '''
    Returning parameter explanation of the model
    Input: model
    Output: string of explanation of hyperparameter involved
    '''
    param_exp = model.extractParamMap()
    param_exp = [(param_exp,)]
    df_param_exp = spark.createDataFrame(param_exp, ["Parameter explanation of the model"])
    return df_param_exp
## =============================================================================
## Test and examples
## =============================================================================
#print()
#print("Data========================")
#df = spark.createDataFrame([[1, Vectors.dense([0.0, 1.0])],
#                             [2, SparseVector(2, {0: 1.0})],], ["id", "features"])
#
#df.show()
#df2 = spark.createDataFrame([[1, Vectors.dense([1000.0, 100.0])],
#                             [2, SparseVector(20, {0: 10.0})],], ["id", "features"])
#
#df2.show()
##lda = LDA(k=2, seed=1, optimizer="em")
##trained_model = lda.fit(df)
#print()
#print("0. config========================")
#
#config = {'k':3, 
#          'maxIter':10, 
#          'seed':-1,
#          'optimizer':'em',
#          'topicConcentration':1.3,
#          'checkpointInterval':2,
#          'learningOffset':0.1,
#          'optimizeDocConcentration':True,
#          'keepLastCheckpoint':False}
#print(config)
#
#print()
#print("1. hiperAdapter========================")
#hiperparameter = hiperAdapter(config)
#print(hiperparameter)
#
#print()
#print("2. train========================")
#trained_model = train(df, hiperparameter)
#print(trained_model)
#
#print("3. copyModel========================")
#cp = copyModel(trained_model)
#
#print("4. topicsDescription========================")
#print(topicsDescription(trained_model).show())
#
#print("5. docConcentrationEstimation========================")
#print(docConcentrationEstimation(trained_model).show())
#
#print("6. paramsExplanation========================")
#print(paramsExplanation(trained_model).show())
#
#print('7. save============')
##saveModel(trained_model,'lda03')
#
#print('8. load============')
#loaded_model = loadModel(trained_model,'lda2')
#loaded_model03_with_loaded02 = loadModel(loaded_model,'lda03')
#
#print('9. modelLikelihood============')
#print(modelLikelihood(loaded_model,df).show())
#print(modelLikelihood(loaded_model03_with_loaded02,df).show())
#
#print('10. modelPerplexity============')
#print(modelPerplexity(loaded_model,df).show())
##print("====================")
##print(type(modelPerplexity(loaded_model03_with_loaded02,df)))
#perp = modelPerplexity(loaded_model03_with_loaded02,df)
#perp.show()
#
#print('11. logPriorModel============')
#print(logPriorModel(loaded_model).show())
#print(logPriorModel(loaded_model03_with_loaded02).show())
#
#print('12. topicsMatrixModel============')
#print(topicsMatrixModel(loaded_model).show())
#print(topicsMatrixModel(loaded_model03_with_loaded02).show())
#
#print('13. trainingLogLikelihoodModel============')
#print(trainingLogLikelihoodModel(loaded_model).show())
#print(trainingLogLikelihoodModel(loaded_model03_with_loaded02).show())
#
#print('14. predict============')
#print(predictDF(loaded_model,df).show())
#print(predictDF(loaded_model03_with_loaded02,df).show())
#
#print('15. vocabSizeModel============')
#print(vocabSizeModel(loaded_model).show())
#print(vocabSizeModel(loaded_model03_with_loaded02).show())
#
#
#print("========finish")

# =============================================================================
# params
# =============================================================================
'''
Param(parent='LDA_457aa537874f9af76e24', 
    name='checkpointInterval', 
    doc='set checkpoint interval (>= 1) or 
    disable checkpoint (-1). E.g. 10 means that 
    the cache will get checkpointed every 10 iterations. 
    Note: this setting will be ignored 
    if the checkpoint directory 
    is not set in the SparkContext')
Param(parent='LDA_457aa537874f9af76e24', 
    name='docConcentration', 
    doc='Concentration parameter (commonly named "alpha") 
    for the prior placed 
    on documents\' distributions over topics ("theta").')
Param(parent='LDA_457aa537874f9af76e24', 
      name='featuresCol', 
      doc='features column name')
Param(parent='LDA_457aa537874f9af76e24', 
      name='k', 
      doc='The number of topics (clusters) to infer. 
      Must be > 1.')
Param(parent='LDA_457aa537874f9af76e24', 
     name='keepLastCheckpoint', 
     doc='(For EM optimizer) If using checkpointing, 
     this indicates whether to keep the last checkpoint. 
     If false, then the checkpoint will be deleted. 
     Deleting the checkpoint can cause failures 
     if a data partition is lost, 
     so set this bit with care.')
Param(parent='LDA_457aa537874f9af76e24', 
    name='learningDecay', 
    doc='(For online optimizer) 
    Learning rate, 
    set as an exponential decay rate. 
    This should be 
    between (0.5, 1.0] 
    to guarantee asymptotic convergence.')
Param(parent='LDA_457aa537874f9af76e24', 
    name='learningOffset', 
    doc='(For online optimizer) A (positive)
    learning parameter 
    that downweights early iterations. 
    Larger values make early iterations count less.')
Param(parent='LDA_4d608fe9f61209df5830', 
    name='maxIter', 
    doc='maximum number of iterations (>= 0)')
Param(parent='LDA_4598a2ed004927661f1c', 
    name='optimizeDocConcentration', 
    doc='(For online optimizer only, currently) 
    Indicates whether the docConcentration 
    (Dirichlet parameter for document-topic distribution) 
    will be optimized during training.')
Param(parent='LDA_4598a2ed004927661f1c', 
    name='optimizer', 
    doc='Optimizer or inference algorithm used to estimate the LDA model. 
    Supported: online, em')
Param(parent='LDA_4598a2ed004927661f1c', 
    name='seed', 
    doc='random seed')
Param(parent='LDA_4598a2ed004927661f1c', 
    name='subsamplingRate', 
    doc='(For online optimizer) Fraction of the corpus 
    to be sampled and used in each iteration 
    of mini-batch gradient descent, 
    in range (0, 1].')
Param(parent='LDA_4598a2ed004927661f1c', 
    name='topicConcentration', 
    doc='Concentration parameter (commonly named "beta" or "eta") 
    for the prior placed on topic\' 
    distributions over terms.')
Param(parent='LDA_4598a2ed004927661f1c', 
    name='topicDistributionCol', 
    doc='Output column with estimates of the topic mixture distribution 
    for each document (often called "theta" in the literature).  
    Returns a vector of zeros for an empty document.')

**catatan: trained_model.getCheckpointFiles() undefined yet
resources:
    http://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf
    http://spark.apache.org/docs/2.2.0/api/python/pyspark.ml.html#pyspark.ml.clustering.LDA
'''