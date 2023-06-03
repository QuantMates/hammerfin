import logging

import pandas as pd

logger = logging.getLogger(__name__)


class Scaler:
    """
    Scaler class for normalizing pandas DataFrame.

    Parameters
    ----------
    method : str or dict, optional
        The method to use for scaling - "standard" or "minmax", by default "standard".
        If a dict is provided, it should be of the form `{"column_name": "method"}`.
        Any column not included will be scaled using the `default_method`.
        `method`has priority over `default_method`.
    default_method: str, optional
        The default method to use for scaling if not provided in the `method` dict, by default "standard"
    skip : list of str, optional
        List of column names to skip while scaling, by default None
    """

    def __init__(self, method="standard", default_method="standard", skip=None):
        self.method = method
        self.default_method = default_method
        self.skip = skip or []
        self.params = {}

    def __str__(self):
        return f"Scaler(method={self.method}, default_method={self.default_method}, skip={self.skip})"

    def __repr__(self):
        return f"Scaler(method={self.method}, default_method={self.default_method}, skip={self.skip})"

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
            method = self.method.get(col, self.default_method) if isinstance(self.method, dict) else self.method
            if method == "standard":
                self.params[col] = {"method": "standard", "mean": X[col].mean(), "std": X[col].std()}
            elif method == "minmax":
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
            method = self.method.get(col, self.default_method) if isinstance(self.method, dict) else self.method
            if method == "standard":
                if self.params[col]["std"] != 0:
                    X[col] = (X[col] - self.params[col]["mean"]) / self.params[col]["std"]
                else:
                    logger.warning(
                        f"HammerFin - Scaler - Column '{col}' had standard deviation of 0 during `fit`. "
                        f"This column will not be changed."
                    )
            elif method == "minmax":
                if self.params[col]["max"] != self.params[col]["min"]:
                    X[col] = (X[col] - self.params[col]["min"]) / (self.params[col]["max"] - self.params[col]["min"])
                else:
                    logger.warning(
                        f"HammerFin - Scaler - Column '{col}' had equal max and min values during `fit`. "
                        f"This column will not be changed."
                    )
        return X
