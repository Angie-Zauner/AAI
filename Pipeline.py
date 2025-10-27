import DiscreteStructureLearning
import Inference
import KBinsDiscretizer
import pandas as ps
import numpy as np
import bnlearn as bn

def BayesianNetworkPipeline(train_set, test_set, cont_columns, nbins, method, scoring_f, max_iter, naive=False)
    
    # --------------------------
    # STEP 0: DISCRETIZATION

    # Fit the discretizer on the training set using cont_columns and nbins
    discretizer, train = discretize_fit(train_set, cont_columns, nbins, transform=True)

    # Trasnsform the test set
    test = discretize_transform(test_set, cont_columns, discretizer)


    # --------------------------
    # STEP 1: STRUCTURE LEARNING/NAIVE BAYES

    if naive: 
        # Define edges 
        features = [col for col in train.columns if col != 'target']
        edges = [(f, 'target') for f in features]

        # Make DAG
        structure = bn.make_DAG(edges)

    else: 
        # Learn structure
        structure = structure_learning(train, method, scoring_f, max_iter, visualise_structure=False)


    # --------------------------
    # STEP 2: PARAMETER LEARNING

    # Learn parameters
    model = bn.parameter_learning.fit(structure, train)


    # --------------------------
    # STEP 3: INFERENCE
    if naive: 
        metrics = infer_and_evaluate(test, model, model_name='NaiveBayes')

    else:
        metrics = infer_and_evaluate(test, model, model_name='BayesianNetwork')

    
    return metrics