import pandas as pd
from sklearn.preprocessing import KBinsDiscretizer

def discretize_fit(df, columns, nbins, transform = False):
    """
    Fit a KBinsDiscretizer to the specified columns of the DataFrame using quantile.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    columns (list): List of column names to discretize.
    n_bins (list): List of number of bins for each column to discretize.

    Returns:
    KBinsDiscretizer: The fitted discretizer.
    """

    # VALIDATION CHECKS: 
    # Columns is a non-empty list 
    if columns is None or not isinstance(columns, (list, tuple)) or len(columns) == 0:
        raise ValueError("`columns` must be a non-empty list of column names.")
    
    # All columns exist in df
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {missing}")
    
    # All columns are numeric
    non_numeric = [c for c in columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"The following columns are not numeric: {non_numeric}")
    
    # No NaN
    if df[columns].isna().any().any():
        raise ValueError("There are NaN values in the columns to discretize. Please impute or remove them first.")
    
    # Number of bins matches number of columns to discretize
    if not isinstance(nbins, (list, tuple)) or len(nbins) != len(columns):
        raise ValueError("`nbins` must be a list or tuple with the same length as `columns`.")
    

    # FIT THE DISCRETIZER
    discretizer = KBinsDiscretizer(n_bins=nbins, encode='ordinal', strategy='quantile')
    discretizer.fit(df[columns])

    if transform == True:
        return discretizer, discretize_transform(df, columns, discretizer)

    return discretizer


def discretize_transform(df, columns, discretizer):
    """
    Transform the specified columns of the DataFrame using the fitted KBinsDiscretizer.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    columns (list): List of column names to discretize.
    discretizer (KBinsDiscretizer): The fitted discretizer.

    Returns:
    pd.DataFrame: The DataFrame with discretized columns.
    """

    # VALIDATION CHECKS:
    # Columns is a non-empty list 
    if columns is None or not isinstance(columns, (list, tuple)) or len(columns) == 0:
        raise ValueError("`columns` must be a non-empty list of column names.")
    
    # All columns exist in df
    missing = [c for c in columns if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in DataFrame: {missing}")
    
    # All columns are numeric
    non_numeric = [c for c in columns if not pd.api.types.is_numeric_dtype(df[c])]
    if non_numeric:
        raise ValueError(f"The following columns are not numeric: {non_numeric}")
    
    # No NaN
    if df[columns].isna().any().any():
        raise ValueError("There are NaN values in the columns to discretize. Please impute or remove them first.")

    # TRANSFORM THE DATA
    transformed = discretizer.transform(df[columns])
    df_transformed = df.copy()
    for i, col in enumerate(columns):
        df_transformed[col] = transformed[:, i]
    
    return df_transformed
