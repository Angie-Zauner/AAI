from collections import defaultdict
import os

def generate_config(structure, name='config', filename=None):
    """
    Generates a configuration file for a bayesian net.

    Parameters

    structure : dict
        Dictionary containing the structure of the Bayesian Network.

    name : str, optional, default='config'
        Name of the configuration (used as the header and default filename).

    filename : str, optional
        Name of the output file. If not provided, defaults to '{name}.txt'.
    """

    if not filename:
        filename = f"{name}.txt"

    # Create 'config' folder if necessary 
    config_dir = os.path.join(os.getcwd(), "config")
    os.makedirs(config_dir, exist_ok=True)

    # complete filepath 
    file_path = os.path.join(config_dir, filename)

    # Extract edges
    if 'model_edges' in structure:
        edges = structure['model_edges']
    else:
        raise KeyError("Edges non trovati in structure_result")

    # Extract parents for each node
    parents = defaultdict(list)
    for parent, child in edges:
        parents[child].append(parent)

    # Find all nodes
    nodes = sorted(set(structure['model'].nodes()))

    # Create random_variables section
    random_variables = []
    for i, node in enumerate(nodes[:-1]):
        if node != 'traget':
            random_variables.append(f"X{i}({node})")
    # Add target
    random_variables.append(f"Y(target)")

    # Create structure section
    structure_lines = []

    # If root write P(node) else write P(node|p1,...,pn)
    for node in nodes:
        if node not in parents:
            structure_lines.append(f"P({node});")
        else:
            structure_lines.append(f"P({node}|{','.join(parents[node])});")

    # Create final text for file
    config_text = (
        "name:" + name + "\n\n"
        "random_variables:" + ";".join(random_variables) + "\n\n"
        "structure:" + "".join(structure_lines)
    )

    # Save
    with open(file_path, "w") as f:
        f.write(config_text)

    print(f"Generated configuration file at: {os.path.abspath(file_path)}")
