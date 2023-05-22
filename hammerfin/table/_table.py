import pandas as pd

from ..dtypes._currency import assert_currency_dtype
from ..dtypes._fin_dtype import FinDType, assert_fin_dtype
from ._financial_methods import (
    calmar,
    cumulative,
    drawdowns,
    indicators,
    max_drawdown,
    sharpe,
    sortino,
)


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


class Table:
    """Overloaded pd.DataFrame class"""
