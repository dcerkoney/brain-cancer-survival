import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, OneToOneFeatureMixin, TransformerMixin


class DataFrameTransformer(BaseEstimator, TransformerMixin, OneToOneFeatureMixin):
    """
    Unfortunately, SimpleImputer() returns a DataFrame that strips the
    categorical dtypes from the columns. This class can be placed after
    SimpleImputer() in a pipeline so that later steps such as OneHotEncoder()
    can be used without errors.
    """

    def __init__(self):
        pass

    def fit(self, X, y=None):
        self._n_features_out = X.shape[1]
        return self

    def transform(self, X):
        df = pd.DataFrame(X, columns=X.columns)
        for col in df.columns:
            df[col] = df[col].astype("category")
        return df

    def get_feature_names_out(self, input_features=None):
        return input_features


class DropCollinearFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=1.0, verbose=False):
        self.threshold = threshold
        self.to_drop_ = None
        self.verbose = verbose

    def fit(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            corr_matrix = X.corr().abs()
        else:
            corr_matrix = pd.DataFrame(X).corr().abs()

        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )
        self.to_drop_ = [
            column
            for column in upper_tri.columns
            if any(upper_tri[column] >= self.threshold)
        ]

        if self.verbose:
            print("Columns dropped due to colinearity:")
            for col in self.to_drop_:
                print(f"  {col}")
            print()

        return self

    def transform(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            return X.drop(columns=self.to_drop_, errors="ignore")
        else:
            return pd.DataFrame(X).drop(columns=self.to_drop_, errors="ignore").values

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array([])
        return np.array([col for col in input_features if col not in self.to_drop_])


class DropLowVarianceConditionedOnOutcome(BaseEstimator, TransformerMixin):
    def __init__(self, threshold=0.01, verbose=False):
        self.threshold = threshold
        self.to_drop_ = None
        self.verbose = verbose

    def fit(self, X, y=None):
        if y is None:
            raise ValueError("y (the outcome variable) must be provided.")

        # Assuming y is a DataFrame with columns 'event' and 'time'
        event = y["Event indicator"]

        self.to_drop_ = []
        for col in X.columns:
            var_alive = X[col][event == 0].var()
            var_deceased = X[col][event == 1].var()
            if var_alive < self.threshold or var_deceased < self.threshold:
                self.to_drop_.append(col)

        if self.verbose:
            print("Columns dropped due to low variance conditioned on outcome:")
            for col in self.to_drop_:
                print(f"  {col}")
            print()

        return self

    def transform(self, X, y=None):
        if isinstance(X, pd.DataFrame):
            return X.drop(columns=self.to_drop_, errors="ignore")
        else:
            return pd.DataFrame(X).drop(columns=self.to_drop_, errors="ignore")

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array([])
        return np.array([col for col in input_features if col not in self.to_drop_])
