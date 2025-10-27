import DiscreteStructureLearning
import Inference
import KbinsDiscretizer
import pandas as ps
import numpy as np
import bnlearn as bn

def BayesianNetworkPipeline(train_set, test_set, cont_columns, nbins,  naive=False):
    
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
        Structure learning method (e.g., 'hc', 'pc').
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
    # STEP 0: DISCRETIZATION

    # Fit the discretizer on the training set using cont_columns and nbins
    discretizer, train = discretize_fit(train_set, cont_columns, nbins, transform=True)

    # Trasnsform the test set
    test = discretize_transform(test_set, cont_columns, discretizer)


    # STEP 1:

    if naive: 

        # Define edges 
        features = [col for col in train_set.columns if col != 'target']
        edges = [(f, 'target') for f in features]

        # Make DAG
        structure = bn.make_DAG(edges)

    else: 
        structure = structure_learning(train_set, "hc", "bic", 2000000, dataset_name="Heart", visualise_structure=False)

    