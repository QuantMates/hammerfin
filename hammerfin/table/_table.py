import logging
import re

import pandas as pd

from ..dtypes._currency import assert_currency_dtype
from ..dtypes._fin_dtype import FinDType, assert_fin_dtype
from ..processing import OneHotEncoder, Scaler
from ..utils.pandas_loader import load_data
from ._financial_methods import (
    calmar,
    cumulative,
    drawdowns,
    indicators,
    max_drawdown,
    sharpe,
    sortino,
)

logger = logging.getLogger(__name__)


def assert_hf(func):
    """Assert that the method is applied to a Table or TableSeries object"""

    def wrapper(*args, **kwargs):
        """Wrapper function for assert_hf"""
        if not isinstance(args[0], Table) and not isinstance(args[0], TableSeries):
            raise ValueError("Method can only be applied to a Table or TableSeries object")
        return func(*args, **kwargs)

    return wrapper


class TableSeries(pd.Series):
    """Overloaded pd.Series class"""

    fin_dtype = None

    def __init__(self, *args, **kwargs):
        if "dtype" in kwargs:
            if isinstance(kwargs["dtype"], FinDType):
                self.fin_dtype = kwargs["dtype"]
                kwargs["dtype"] = self.fin_dtype.numpy_dtype
        super().__init__(*args, **kwargs)

    @property
    def _constructor(self):
        return TableSeries

    @property
    def _constructor_expanddim(self):
        return Table

    def __str__(self):
        if self.fin_dtype:
            return pd.Series(self).__str__() + f"\nfin-dtype: {self.fin_dtype}"
        return pd.Series(self).__str__()

    def __repr__(self):
        if self.fin_dtype:
            return pd.Series(self).__repr__() + f"\nfin-dtype: {self.fin_dtype}"
        return pd.Series(self).__repr__()

    @assert_fin_dtype
    def is_fin_dtype(self):
        """Return True if the TableSeries has a fin_dtype"""
        return True

    @assert_currency_dtype
    def is_currency_dtype(self):
        """Return True if the TableSeries is a Currency fin_dtype"""
        return True

    sharpe = assert_hf(sharpe)
    calmar = assert_hf(calmar)
    sortino = assert_hf(sortino)
    drawdowns = assert_hf(drawdowns)
    max_drawdown = assert_hf(max_drawdown)
    indicators = assert_hf(indicators)
    cumulative = assert_hf(cumulative)

    def as_df(self):
        """Return the TableSeries as a pandas Series"""
        return pd.Series(self)


class Table(pd.DataFrame):
    """Overloaded pd.DataFrame class"""

    _metadata = ["processing_steps"]

    @property
    def _constructor(self):
        return Table

    @property
    def _constructor_sliced(self):
        return TableSeries

    def __setitem__(self, key, value):
        if isinstance(value, TableSeries) and value.fin_dtype is not None:
            dtype_dict = self.__dict__.get("_tableseries_dtypes", {})
            dtype_dict[key] = value.fin_dtype
            self.__dict__["_tableseries_dtypes"] = dtype_dict
        super().__setitem__(key, value)

    def __getitem__(self, item):
        value = super().__getitem__(item)
        if isinstance(value, pd.Series):
            fin_dtype = self.__dict__.get("_tableseries_dtypes", {}).get(item)
            if fin_dtype is not None:
                value = TableSeries(value, dtype=fin_dtype)
        return value

    def __init__(
        self,
        data=None,
        copy=False,
        verbose=False,
        **kwargs,
    ):
        self.processing_steps = []
        if isinstance(data, str):
            data = load_data(data, **kwargs)
            super().__init__(data)
        else:
            super().__init__(data=data, copy=copy, **kwargs)

        self.is_stationnary = False
        self.is_normalized = False
        nb_rows, nb_cols = self.shape

        if verbose:
            logger.info(f"Table has {nb_rows} rows and {nb_cols} columns")
            logger.info(f"Table has {self.memory_usage().sum()/1e6} MB of memory")
            logger.info(f"Table columns are {[*self.columns]}")
            nb_nan = self.isna().sum().sum()
            if nb_nan > 0:
                logger.warning(f"Table contains {nb_nan} NaN values")
            else:
                logger.info("Table has no NaN values")

    def __type__(self):
        return "Table"

    def __repr__(self):
        return super().__repr__() + " Table Object"

    def __str__(self):
        return super().__str__() + " Table Object"

    def find_date(self):
        """Find the datetime column and set it as index"""
        if not pd.api.types.is_datetime64_any_dtype(self.index):
            found = False
            for col in self.columns:
                first_value = self[col][0]
                if isinstance(first_value, str):
                    if re.match(r"\d{4}-\d{2}-\d{2}", first_value) or re.match(r"\d{2}/\d{2}/\d{4}", first_value):
                        self[col] = pd.to_datetime(self[col])
                        self.set_index(col, inplace=True)
                        logger.info(f"Column {col} is set as index")
                        found = True
                        break
                elif isinstance(first_value, int):
                    if first_value > 1e9:
                        self[col] = pd.to_datetime(self[col], unit="s")
                        self.set_index(col, inplace=True)
                        logger.info(f"Column {col} is set as index")
                        found = True
                        break

                elif pd.api.types.is_datetime64_any_dtype(first_value):
                    self.set_index(col, inplace=True)
                    logger.info(f"Column {col} is set as index")
                    found = True
                    break
            self.index.name = "date"
            if not found:
                logger.info("No datetime column found")
        return self

    sharpe = assert_hf(sharpe)
    calmar = assert_hf(calmar)
    sortino = assert_hf(sortino)
    drawdowns = assert_hf(drawdowns)
    max_drawdown = assert_hf(max_drawdown)
    indicators = assert_hf(indicators)
    cumulative = assert_hf(cumulative)

    def apply_processing(self, X):
        """Apply processing to another Table"""
        X = X.copy()
        for step in self.processing_steps:
            X = step.transform(X)
        return X

    def scale(self, *args, **kwargs):
        """Scale the Table"""
        scaler = Scaler(*args, **kwargs).fit(self)
        self.processing_steps.append(scaler)
        transformed = scaler.transform(self)
        self.update(transformed)
        return self

    def one_hot_encode(self, *args, **kwargs):
        """One-hot-encode the Table"""
        one_hot_encoder = OneHotEncoder(*args, **kwargs).fit(self)
        self.processing_steps.append(one_hot_encoder)
        transformed = one_hot_encoder.transform(self)
        self.update(transformed)
        return self
