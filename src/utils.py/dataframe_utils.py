import pandas as pd


def get_dataframe_unique_values_per_column(df: pd.DataFrame) -> pd.Series:
    '''
    Gets all of the unique values within each column of the input DataFrame.

    Parameters:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.Series: Series containing the DataFrame column names as the index
            and a list of all of the unique values of that column as the values.
    '''
    return pd.Series({c: df[c].unique() for c in df})