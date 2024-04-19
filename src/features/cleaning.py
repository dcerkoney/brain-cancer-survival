from typing import Tuple

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
          .pipe(clean_year_of_death_recode)
          .pipe(clean_age_recode_with_lt1_year_olds)
          .pipe(clean_median_household_income)
          .pipe(select_survival_months_flag)
          .pipe(convert_columns_to_categorical)
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

def clean_year_of_death_recode(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Converts the "Year of death recode" column to an int and creates a separate 
    "Alive at last contact" column.

    Parameters:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Transformed DataFrame.
    '''
    df['Alive at last contact'] = (df['Year of death recode'] == 'Alive at last contact')
    df['Year of death recode'] = (
        df['Year of death recode']
        .replace({'Alive at last contact': np.nan})
        .astype('Int64')
    )
    return df

def clean_age_recode_with_lt1_year_olds(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Converts each age range to the first year of that range.

    Parameters:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Transformed DataFrame.
    '''
    df['Age recode with <1 year olds'] = (
        df['Age recode with <1 year olds']
        .str.slice(0, 2)
        .astype(int)
    )
    return df

def clean_median_household_income(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Converts each median household income range to the lower limit of that 
    range (or a NaN).

    Parameters:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Transformed DataFrame.
    '''
    df['Median household income inflation adj to 2021 (thousands USD)'] = (
        df['Median household income inflation adj to 2021']
        .str.slice(1, 3)
        .replace({
            'nk': np.nan,  # 'Unknown/missing/no match/Not 1990-2021' -> NaN
            ' $': '18'     # '< $35,000' -> 18 
        })
        .astype('Int64')
    )
    df = df.drop(columns='Median household income inflation adj to 2021')
    return df

def select_survival_months_flag(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Selects patients with complete date information available in the SEER*Stat
    database and non-zero days of survival based on the "Survival months flag" 
    variable.

    Parameters:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Transformed DataFrame.
    '''
    val = 'Complete dates are available and there are more than 0 days of survival'
    return df[(df['Survival months flag'] == val)]

def convert_columns_to_categorical(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Converts columns containing categorical data to actually have categorical
    dtypes in the DataFrame. 
    
    Note: The list of categorical columns used in this function is not 
    necessarily complete.

    Parameters:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: DataFrame with dtypes changed.
    '''
    categorical_columns = [
        'Sex',
        'Race recode (W, B, AI, API)', 
        'Origin recode NHIA (Hispanic, Non-Hisp)', 
        'Race and origin recode (NHW, NHB, NHAIAN, NHAPI, Hispanic)', 
        'Vital status recode (study cutoff used)', 
        'SEER cause-specific death classification', 
        'SEER other cause of death classification', 
        'Type of Reporting Source', 
        'Marital status at diagnosis', 
        'Rural-Urban Continuum Code', 
        'End Calc Vital Status (Adjusted)', 
        'Survival months flag'
    ]
    return df.astype({col: 'category' for col in categorical_columns})

def split_X_and_y_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    '''
    Splits the data into independent and dependent variables in the format 
    needed for model training.

    Parameters:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A tuple containting two DataFrame's, 
            one with the X variables and one with the y variables.
    '''
    X = df.drop(
        ['Vital status recode (study cutoff used)', 'Survival months'],
        axis='columns'
    )
    y = pd.DataFrame({
        'Event indicator': 
            (df['Vital status recode (study cutoff used)'] == 'Dead'),
        'Survival months': 
            df['Survival months']
    })
    return (X, y)