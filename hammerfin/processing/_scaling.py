import pandas as pd


# pylint: disable=fixme
# TODO: take method as a dict of column names and methods
class Scaler:
    """
    Scaler class for normalizing pandas DataFrame.

    Parameters
    ----------
    method : str, optional
        The method to use for scaling - "standard" or "minmax", by default "standard"
    skip : list of str, optional
        List of column names to skip while scaling, by default None
    """

    def __init__(self, method="standard", skip=None):
        self.method = method
        self.skip = skip or []
        self.params = {}

    def __str__(self):
        return f"Scaler(method={self.method}, skip={self.skip})"

    def __repr__(self):
        return f"Scaler(method={self.method}, skip={self.skip})"

    def fit(self, X):
        """
        Fit the Scaler to the data.

        Parameters
        ----------
        X : pd.DataFrame
            A pandas DataFrame to fit the Scaler
        """
        for col in X.columns:
            if col in self.skip:
                continue
            if not pd.api.types.is_numeric_dtype(X[col]):
                self.skip.append(col)
                continue
            if self.method == "standard":
                self.params[col] = {"method": "standard", "mean": X[col].mean(), "std": X[col].std()}
            elif self.method == "minmax":
                self.params[col] = {"method": "minmax", "min": X[col].min(), "max": X[col].max()}
            else:
                raise ValueError("Scaler method must be 'standard' or 'minmax'")
        return self

    def transform(self, X):
        """
        Transform the data using the fitted Scaler.

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
        for col in X.columns:
            if col in self.skip:
                continue
            if self.method == "standard":
                if self.params[col]["std"] != 0:
                    X[col] = (X[col] - self.params[col]["mean"]) / self.params[col]["std"]
            elif self.method == "minmax":
                if self.params[col]["max"] != self.params[col]["min"]:
                    X[col] = (X[col] - self.params[col]["min"]) / (self.params[col]["max"] - self.params[col]["min"])
        return X
