import bnlearn as bn
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt


def save_structure(edges, title, file_name):
    G = nx.DiGraph()
    G.add_edges_from(edges)
    pos = nx.spring_layout(G)
    plt.figure(figsize=(8, 6))
    nx.draw(G, pos, with_labels=True, node_size=1000, node_color='lightblue', font_size=10, arrows=True)
    plt.title(title)
    plt.savefig(file_name, dpi=300, bbox_inches='tight')
    print("DAG for %s available at %s\n" % (title, file_name))

def structure_learning(training_data, method_type, scoring_function, max_iterations, visualise_structure=True, dataset_name="dataset"):
    
    model = bn.structure_learning.fit(training_data, methodtype=method_type, scoretype=scoring_function, max_iter=max_iterations)
    print("model [%s]=%s" % (method_type, model))
    print("num_model_edges [%s]=%s" % (method_type, len(model['model_edges'])))

    # visualise the learnt structure
    if visualise_structure:
        title = "Learnt Structure %s" % (method_type)
        save_structure(model['model_edges'], title, "structures/%s-%s-%s-DAG.png" % (dataset_name, method_type, scoring_function))
    
    return model