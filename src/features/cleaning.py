import numpy as np
import pandas as pd


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Performs basic cleaning of the data. More advanced cleaning and feature 
    selection steps are performed with scikit-learn.

    Parameters:
        df (pd.DataFrame): Raw DataFrame containing the data.

    Returns:
        pd.DataFrame: The resulting cleaned DataFrame.
    '''
    return (
        df.pipe(replace_empty_values)
          .pipe(remove_empty_columns)
    )

def replace_empty_values(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Replaces strings that represent missing data with NaNs.

    Parameters:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Transformed DataFrame.
    '''
    strings_to_replace = ['Blank(s)', 'Recode not available', 'Unclassified']
    replacements = {value: np.nan for value in strings_to_replace}
    return df.replace(replacements)

def remove_empty_columns(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Drops columns with no useful information. These can be:
      1. columns of all NaNs
      2. columns with no unique values

    Parameters:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Transformed DataFrame.
    '''
    df = df.dropna(axis='columns', how='all')
    return df[[c for c in list(df) if len(df[c].unique()) > 1]]