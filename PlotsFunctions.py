import matplotlib as plt
import seaborn as sns
import pandas as pd
import numpy as np
import math

def plot_distributions(df, columns, palette="Set2"):
    """
    Plot the distributions of the specified columns in the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    columns (list): List of column names to plot.
    """
    num_cols = len(columns)
    num_rows = math.ceil(num_cols / 2)

    plt.figure(figsize=(12, num_rows * 4))

    for i, col in enumerate(columns):
        plt.subplot(num_rows, 2, i + 1)
        sns.histplot(df[col], kde=True, color=sns.color_palette(palette)[i % len(sns.color_palette(palette))])
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')

    plt.tight_layout()
    plt.show()