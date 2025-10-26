import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import math

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