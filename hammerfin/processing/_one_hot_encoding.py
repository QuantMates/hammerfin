import logging

import pandas as pd

logger = logging.getLogger(__name__)


class OneHotEncoder:
    """
    OneHotEncoder class for pandas DataFrame.

    Parameters
    ----------
    max_unique : int, optional
        The maximum number of unique values allowed for a categorical variable.
    skip : list of str, optional
        List of column names to skip while one-hot encoding, by default None
    """

    def __init__(self, max_unique=10, skip=None):
        self.max_unique = max_unique
        self.skip = skip or []
        self.columns = {}

    def __str__(self):
        return f"OneHotEncoder(max_unique={self.max_unique}, skip={self.skip})"

    def __repr__(self):
        return f"OneHotEncoder(max_unique={self.max_unique}, skip={self.skip})"

    def fit(self, X):
        """
        Fit the OneHotEncoder to the data.

        Parameters
        ----------
        X : pd.DataFrame
            A pandas DataFrame to fit the OneHotEncoder
        """
        for col in X.columns:
            if col in self.skip:
                continue
            if not pd.api.types.is_numeric_dtype(X[col]):
                if X[col].nunique() > self.max_unique:
                    raise ValueError(
                        f"Column '{col}' has more than {self.max_unique} unique values. "
                        f"Please drop this column or reduce its unique values."
                    )
                self.columns[col] = X[col].unique().tolist()
        return self

    def transform(self, X):
        """
        Transform the data using the fitted OneHotEncoder.

        Parameters
        ----------
        X : pd.DataFrame
            A pandas DataFrame to transform

        Returns
        -------
        pd.DataFrame
            Transformed pandas DataFrame
        """
        X = X.copy()
        for col, values in self.columns.items():
            for unique_value in X[col].unique().tolist():
                if unique_value not in values:
                    logger.warning(
                        f"HammerFin - OneHotEncoder - Unseen category '{unique_value}' in column '{col}' during 'fit'. "
                        f"This category will be ignored."
                    )
                else:
                    X[f"{col}_{unique_value}"] = (X[col] == unique_value).astype(int)
            X.drop(col, axis=1, inplace=True)
        return X
