import logging
import re

import pandas as pd

from ..dtypes._currency import assert_currency_dtype
from ..dtypes._fin_dtype import FinDType, assert_fin_dtype
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

# pylint: disable=logging-fstring-interpolation, too-many-branches


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

    sharpe = sharpe
    calmar = calmar
    sortino = sortino
    drawdowns = drawdowns
    max_drawdown = max_drawdown
    indicators = indicators
    cumulative = cumulative

    def as_df(self):
        """Return the TableSeries as a pandas Series"""
        return pd.Series(self)


class Table(pd.DataFrame):
    """Overloaded pd.DataFrame class"""

    @property
    def _constructor(self):
        return Table

    @property
    def _constructor_sliced(self):
        return TableSeries

    def __init__(
        self,
        data=None,
        dtype=None,
        copy=False,
        timeseries=True,
        verbose=False,
        parse_dates=False,
        **kwargs,
    ):
        if isinstance(data, str):
            data = load_data(data, **kwargs)
            super().__init__(data)
        else:
            super().__init__(data=data, dtype=dtype, copy=copy, **kwargs)

        self.is_stationnary = False
        self.is_normalized = False
        nb_rows, nb_cols = self.shape

        if verbose:
            print(f"Table has {nb_rows} rows and {nb_cols} columns")
            print(f"Table has {self.memory_usage().sum()/1e6} MB of memory")
            print(f"Table columns are {[*self.columns]}")
            nb_nan = self.isna().sum().sum()
            if nb_nan > 0:
                logger.warning(f"Table contains {nb_nan} NaN values")
            else:
                print("Table has no NaN values")

        if timeseries and not pd.api.types.is_datetime64_any_dtype(self.index) and parse_dates:
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

    def __type__(self):
        return "Table"

    def __repr__(self):
        return super().__repr__() + " Table Object"

    def __str__(self):
        return super().__str__() + " Table Object"

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, **kwargs)

    def as_df(self):
        """Return the Table as a pandas DataFrame"""
        return pd.DataFrame(self)

    sharpe = sharpe
    calmar = calmar
    sortino = sortino
    drawdowns = drawdowns
    max_drawdown = max_drawdown
    indicators = indicators
    cumulative = cumulative
