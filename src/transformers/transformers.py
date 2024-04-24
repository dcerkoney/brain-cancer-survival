import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class DataFrameTransformer(BaseEstimator, TransformerMixin):
    '''
    Unfortunately, SimpleImputer() returns a DataFrame that strips the 
    categorical dtypes from the columns. This class can be placed after 
    SimpleImputer() in a pipeline so that later steps such as OneHotEncoder() 
    can be used without errors.
    '''
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = pd.DataFrame(X, columns=X.columns)
        for col in df.columns:
            df[col] = df[col].astype('category')
        return df