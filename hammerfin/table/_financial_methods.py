import numpy as np
import pandas as pd


def assert_ts(func):
    """Assert that the method is applied to a Table object with datetime index"""

    def wrapper(*args, **kwargs):
        """Wrapper function for assert_ts"""
        # pylint: disable = import-outside-toplevel
        from ._table import Table, TableSeries

        if not isinstance(args[0], Table) and not isinstance(args[0], TableSeries):
            raise ValueError("Method can only be applied to Table object")
        if not pd.api.types.is_datetime64_any_dtype(args[0].index):
            raise ValueError("Method can only be applied to Table with datetime index")
        return func(*args, **kwargs)

    return wrapper


def daily_resampler(self):
    """Resample to daily frequency and return last value"""
    return self.resample("1D").last().fillna(method="ffill")  # if oversampling


@assert_ts
def sharpe(self, start=None, end=None, risk_free=0):
    """
    Calculate the Sharpe ratio for the given Table or TableSeries.

    The Sharpe ratio is a measure of risk-adjusted return. It is calculated
    as the difference between the return of the investment and the risk-free
    rate, divided by the standard deviation of the investment's return.

    Parameters
    ----------
    start : str or datetime-like, optional
        Start date for the period to calculate the Sharpe ratio. If None, the
        start of the data is used. Default is None.
    end : str or datetime-like, optional
        End date for the period to calculate the Sharpe ratio. If None, the
        end of the data is used. Default is None.
    risk_free : float, optional
        The risk-free rate to use in the calculation. Default is 0.

    Returns
    -------
    float
        The Sharpe ratio.

    Raises
    ------
    ValueError
        If the method is not applied to a Table or TableSeries object, or if
        the object does not have a datetime index.
    """

    _daily = daily_resampler(self)
    _E = _daily.loc[start:end].mean() * 252 - risk_free
    _std = _daily.loc[start:end].std(ddof=0) * np.sqrt(252)
    return _E / _std
