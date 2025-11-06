from DiscreteStructureLearning import *
from Inference import *
from KbinsDiscretizer import *
import pandas as ps
import numpy as np
import bnlearn as bn
import time
import os
import sys
import contextlib


def BayesianNetworkPipeline(train_set, test_set, cont_columns, nbins=None, method='hc', scoring='aic', max_iter = 200000, naive=False):
    
    """
    Full Bayesian Network pipeline: discretization, structure learning (or naive),
    parameter learning, and inference evaluation.

    Parameters
    ----------
    train_set : pd.DataFrame
        Training dataset containing the target.
    test_set : pd.DataFrame
        Test dataset containing the target.
    cont_columns : list
        Continuous columns to discretize.
    nbins : int
        Number of bins per column for discretization.
    method : str
        Structure learning method (e.g., 'hc', 'cl').
    scoring_f : str
        Scoring function for structure learning (e.g., 'aic', 'bic').
    max_iter : int
        Max iterations for structure learning.
    naive : bool
        If True, build a Naive Bayes DAG manually.

    Returns
    -------
    metrics : dict
        Evaluation metrics from the inference step.
    """
    # Initialize nbins to be 3 for each column if not specified
    if not nbins:
        nbins = [3 for i in len(cont_columns)]

    #-----------------------
    # STEP 0: DISCRETIZATION
    start_train = time.time()

    # Fit the discretizer on the training set using cont_columns and nbins
    discretizer, train = discretize_fit(train_set, cont_columns, nbins, transform=True)

    # Trasnsform the test set
    test = discretize_transform(test_set, cont_columns, discretizer)


    #---------------------------
    # STEP 1: STRUCTURE LEARNING

    # Naive Bayes
    if naive: 
        # Define edges 
        features = [col for col in train.columns if col != 'target']
        edges = [(f, 'target') for f in features]

        # Make DAG
        structure = bn.make_DAG(edges)

    #BayesianNetwork
    else: 
        structure = structure_learning(train, method, scoring, max_iter, visualise_structure=False)
    
    #---------------------------
    # STEP 2: PARAMETER LEARNING

    # Learn parameters
    model = bn.parameter_learning.fit(structure, train, verbose = 0)

    train_time = time.time() - start_train

    #---------------------------
    # STEP 3: INFERENCE
    
    if naive:
        metrics = infer_and_evaluate(test, model, train_time, model_name='NaiveBayes')

    else:
        metrics = infer_and_evaluate(test, model, train_time, model_name='BayesianNetwork')
    
    return metrics