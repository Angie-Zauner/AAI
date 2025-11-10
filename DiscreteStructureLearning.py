import bnlearn as bn
import pandas as pd
import networkx as nx
import os
import matplotlib.pyplot as plt


def save_structure(edges, title, file_name, color):
    """
    Save and visualize a Bayesian Network structure (DAG) as an image file.

    Parameters

    edges : List of directed edges representing the structure of the Bayesian Network,

    title : Title to display on the plotted graph.

    file_name : Output path where the DAG image will be saved.

    color : Node color in the visualization.
    """
    # Create a directed graph
    G = nx.DiGraph()
    G.add_edges_from(edges)

    # Compute node positions
    pos = nx.spring_layout(G)

    # Plot
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_size=1000, node_color=color, font_size=10, arrows=True)

     # Add title and save to file
    plt.title(title)
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    print("DAG for %s available at %s\n" % (title, file_name))



def structure_learning(training_data, method_type="hc", scoring_function="bic" ,max_iterations=200000, visualise_structure=True, dataset_name="dataset", color='lightskyblue'):
    """
    Learn the structure of a Bayesian Network from data using bnlearn.

    Parameters
    training_data : Dataset used to learn the Bayesian Network structure.
    method_type : str, optional, default='hc'
    scoring_function : str, optional, default='bic'
    max_iterations : Maximum number of iterations for the learning algorithm, optional, default=200000 
    visualise_structure : If True, saves a visual representation of the learned DAG.
    dataset_name : Dataset name used for labeling the output files.
    color : Node color in the DAG visualization.

    Returns
    model : Learned Bayesian Network model object as returned by `bnlearn`.

    """
    model = bn.structure_learning.fit(training_data, methodtype=method_type, scoretype=scoring_function, max_iter=max_iterations, verbose=0)
    print("model [%s]=%s" % (method_type, model))
    print("num_model_edges [%s]=%s" % (method_type, len(model['model_edges'])))

    # visualise the learnt structure
    if visualise_structure:
        title = "Learnt Structure %s" % (method_type)
        
        # Create 'structures' folder if necessary 
        config_dir = os.path.join(os.getcwd(), "structures")
        os.makedirs(config_dir, exist_ok=True)
        
        save_structure(model['model_edges'], title, "structures/%s-%s-%s-DAG.png" % (dataset_name, method_type, scoring_function), color)
    
    return model