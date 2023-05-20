import pandas as pd

from ..dtypes._custom_dtype import CustomDType
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

    custom_dtype = None

    def __init__(self, *args, **kwargs):
        if "dtype" in kwargs:
            if isinstance(kwargs["dtype"], CustomDType):
                self.custom_dtype = kwargs["dtype"]
                kwargs["dtype"] = self.custom_dtype.numpy_dtype
        super().__init__(*args, **kwargs)

    @property
    def _constructor(self):
        return TableSeries

    @property
    def _constructor_expanddim(self):
        return Table

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
