import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, OneToOneFeatureMixin


class DataFrameTransformer(BaseEstimator, TransformerMixin, OneToOneFeatureMixin):
    '''
    Unfortunately, SimpleImputer() returns a DataFrame that strips the 
    categorical dtypes from the columns. This class can be placed after 
    SimpleImputer() in a pipeline so that later steps such as OneHotEncoder() 
    can be used without errors.
    '''
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        self._n_features_out = X.shape[1]
        return self
    
    def transform(self, X):
        df = pd.DataFrame(X, columns=X.columns)
        for col in df.columns:
            df[col] = df[col].astype('category')
        return df
    
    def get_feature_names_out(self, input_features=None):
        return input_features