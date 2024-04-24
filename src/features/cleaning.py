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
          .pipe(clean_tumor_size_codes)
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
    mask_nan = df['Median household income inflation adj to 2021 (thousands USD)'].notna(
    )
    return df[mask_nan]

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

def convert_NaN_to_missing(df: pd.DataFrame, columns):
    '''
    Coverts NaN values to "MISSING" for a set of columns.
    '''
    ...
    
def clean_tumor_size_codes(df: pd.DataFrame) -> pd.DataFrame:
    '''
    Cleans the tumor size code features covering different time periods 
    ("EOD 10 - size (1988-2003)", "CS tumor size (2004-2015)", "Tumor Size 
    Summary (2016+)") in to following ways:
      1. Drops the codes for non-exact size measurements and other special 
         codes. It looks like there are only a few of these (~10's) for each of 
         these codes.
      2. Combines the listed exact size measurements from across time periods 
         into a single feature.
      3. Creates another feature to track no tumor found.
      4. Creates another feature to track missing values.

    Parameters:
        df (pd.DataFrame): Input DataFrame.

    Returns:
        pd.DataFrame: Transformed DataFrame.
    '''
    def drop_tumor_size_codes(df, column, threshold_code):
        '''
        Drops samples with tumor size codes above a threshold except for values
        of 999.
        '''
        mask = np.logical_and(
            df[column] > threshold_code,
            df[column] != 999
        )
        return df.drop(df[mask].index)
    
    df['EOD 10 - size (1988-2003)'] = df['EOD 10 - size (1988-2003)'].astype('Int64')
    df['CS tumor size (2004-2015)'] = df['CS tumor size (2004-2015)'].astype('Int64')
    df['Tumor Size Summary (2016+)']    = df['Tumor Size Summary (2016+)'].astype('Int64')
    
    # Drop rows with the uncommon code
    df = drop_tumor_size_codes(df, 'EOD 10 - size (1988-2003)', 996)
    df = drop_tumor_size_codes(df, 'CS tumor size (2004-2015)', 989)
    df = drop_tumor_size_codes(df, 'Tumor Size Summary (2016+)', 989)
        
    sizes_88_03 = df['EOD 10 - size (1988-2003)'].astype('Int64')
    sizes_04_15 = df['CS tumor size (2004-2015)'].astype('Int64')
    sizes_16    = df['Tumor Size Summary (2016+)'].astype('Int64')
    
    # Create new features for combined tumor size, no tumor found, and missing
    # values
    simplefilter(action='ignore', category=pd.errors.PerformanceWarning)
    df['Combined Tumor Size'] = (
        sizes_88_03
        .combine_first(sizes_04_15)
        .combine_first(sizes_16)
    )
    df['Combined Tumor Size'] = df['Combined Tumor Size'].replace(999, 0)
    df['No tumor found'] = np.logical_or.reduce((
        (sizes_88_03 == 000).fillna(False), 
        (sizes_04_15 == 000).fillna(False), 
        (sizes_16 == 000).fillna(False))
    )
    df['Unknown tumor size'] = np.logical_or.reduce((
        (sizes_88_03 == 999).fillna(False), 
        (sizes_04_15 == 999).fillna(False), 
        (sizes_16 == 999).fillna(False))
    )
    simplefilter(action='default', category=pd.errors.PerformanceWarning)
    
    return df