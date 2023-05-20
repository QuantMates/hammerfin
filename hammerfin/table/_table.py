import pandas as pd

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
