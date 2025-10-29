import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math
import seaborn as sns

def plot_distributions(df, columns, ncol=4, color="lightskyblue"):
    """
    Plot the distributions of the specified columns in the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    columns (list): List of column names to plot.
    ncol (int): Number of subplots per column.
    color (str): Color of the plot.
    """
    num_cols = len(columns)
    num_rows = math.ceil(num_cols / ncol)

    plt.figure(figsize=(ncol*4, num_rows * 4))

    for i, col in enumerate(columns):
        plt.subplot(num_rows, ncol, i + 1)
        plt.hist(df[col], bins=30, color=color, alpha=0.7)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()


def plot_fold_metrics(fold_results, metric_keys=None, palette="tab10"):
    """
    Generates line plots of fold values for each metric.
    
    Parameters
    ----------
    fold_results : dict
        Dictionary with structure:
        fold_results[scoring][metric] = list of values for each fold
    metric_keys : list, optional
        List of metrics to plot. If None, uses all keys from the first scoring.
    """

    # List of scoring methods
    scoring_list = list(fold_results.keys())
    
    # Number of folds (take from the first scoring and first metric)
    if metric_keys is None:
        metric_keys = list(fold_results[scoring_list[0]].keys())
    n_folds = len(next(iter(fold_results.values()))[metric_keys[0]])
    
    # Color palette
    colors = sns.color_palette(palette, colors=len(scoring_list))
    
    # Create a plot for each metric
    for metric in metric_keys:
        plt.figure(figsize=(8,5))
        
        for i, s in enumerate(scoring_list):
            y = fold_results[s][metric]
            x = np.arange(1, n_folds+1)
            plt.plot(x, y, marker='o', color=colors[i], label=s.upper())
        
        plt.title(f"{metric} per fold")
        plt.xlabel("Fold")
        plt.ylabel(metric)
        plt.xticks(x)
        plt.grid(True, linestyle='--', alpha=0.5)
        plt.legend(title="Scoring Method")
        plt.tight_layout()
        plt.show()