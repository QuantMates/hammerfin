import numpy as np
import pandas as pd

__available_indicators__ = ["sharpe", "sortino", "calmar", "max_drawdown"]


def assert_ts(func):
    """Assert that the method is applied to an object with datetime index"""

    def wrapper(*args, **kwargs):
        """Wrapper function for assert_ts"""

        if not pd.api.types.is_datetime64_any_dtype(args[0].index):
            raise ValueError("Method can only be applied to an object with datetime index")
        return func(*args, **kwargs)

    return wrapper


def daily_resampler(self):
    """Resample to daily frequency"""
    print(self.resample("1D").last().fillna(method="ffill"))
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
    """

    _daily = daily_resampler(self)
    _E = _daily.loc[start:end].mean() * 252 - risk_free
    _std = _daily.loc[start:end].std(ddof=0) * np.sqrt(252)
    return _E / _std


@assert_ts
def sortino(self, start=None, end=None, risk_free=0):
    """
    Calculate the Sortino ratio for the given Table or TableSeries.

    The Sortino ratio is a variation of the Sharpe ratio that differentiates harmful
    volatility from total overall volatility by using the standard deviation of negative
    asset returns, called downside deviation.

    Parameters
    ----------
    start : str or datetime-like, optional
        Start date for the period to calculate the Sortino ratio. If None, the
        start of the data is used. Default is None.
    end : str or datetime-like, optional
        End date for the period to calculate the Sortino ratio. If None, the
        end of the data is used. Default is None.
    risk_free : float, optional
        The risk-free rate to use in the calculation. Default is 0.

    Returns
    -------
    float
        The Sortino ratio.
    """
    _daily = daily_resampler(self)
    _E = _daily.loc[start:end].mean() * 252 - risk_free
    _std_neg = _daily[_daily < 0].loc[start:end].std(ddof=0) * np.sqrt(252)
    return _E / _std_neg


@assert_ts
def drawdowns(self, start=None, end=None):
    """
    Calculate the drawdowns for the given Table or TableSeries.

    Drawdown is the measure of the decline from a historical peak in some variable
    (typically the cumulative profit or total open equity of a financial trading strategy).

    Parameters
    ----------
    start : str or datetime-like, optional
        Start date for the period to calculate the drawdowns. If None, the
        start of the data is used. Default is None.
    end : str or datetime-like, optional
        End date for the period to calculate the drawdowns. If None, the
        end of the data is used. Default is None.

    Returns
    -------
    pandas.Series
        The drawdowns.
    """

    _daily = daily_resampler(self)
    _price = (_daily.loc[start:end] + 1).cumprod() - 1
    _cummax = _price.cummax()
    return _cummax - _price


@assert_ts
def max_drawdown(self, start=None, end=None, risk_free=0):  # pylint: disable = unused-argument
    """
    Calculate the maximum drawdown for the given Table or TableSeries.

    Maximum drawdown (MDD) is an indicator of the risk of a portfolio selected based on a
    certain strategy. It measures the largest single drop from peak to bottom in the value
    of a portfolio (before a new peak is achieved).

    Parameters
    ----------
    start : str or datetime-like, optional
        Start date for the period to calculate the maximum drawdown. If None, the
        start of the data is used. Default is None.
    end : str or datetime-like, optional
        End date for the period to calculate the maximum drawdown. If None, the
        end of the data is used. Default is None.
    risk_free : float, optional
        The risk-free rate to use in the calculation. Default is 0.

    Returns
    -------
    float
        The maximum drawdown.
    """
    return self.drawdowns(start, end).max()


@assert_ts
def calmar(self, start=None, end=None, risk_free=0):
    """
    Calculate the Calmar ratio for the given Table or TableSeries.

    The Calmar ratio is a performance measurement that compares the average annual
    compounded rate of return and the maximum drawdown risk of commodity trading advisors
    and hedge funds. The lower the Calmar ratio, the worse the investment performed
    on a risk-adjusted basis over the specified time period; the higher the Calmar ratio,
    the better it performed.

    Parameters
    ----------
    start : str or datetime-like, optional
        Start date for the period to calculate the Calmar ratio. If None, the
        start of the data is used. Default is None.
    end : str or datetime-like, optional
        End date for the period to calculate the Calmar ratio. If None, the
        end of the data is used. Default is None.
    risk_free : float, optional
        The risk-free rate to use in the calculation. Default is 0.

    Returns
    -------
    float
        The Calmar ratio.
    """
    # not annualized
    _daily = daily_resampler(self)
    _E = _daily.loc[start:end].mean() - risk_free
    _max_drawdown = abs(self.max_drawdown(start, end))
    return _E / _max_drawdown


@assert_ts
def indicators(self, start=None, end=None, risk_free=0):
    """
    Calculate the financial indicators for the given Table or TableSeries.

    This method calculates a set of financial indicators for the data, such as
    the Sharpe ratio, Sortino ratio, drawdowns, maximum drawdown, and Calmar ratio.

    Parameters
    ----------
    start : str or datetime-like, optional
        Start date for the period to calculate the indicators. If None, the
        start of the data is used. Default is None.
    end : str or datetime-like, optional
        End date for the period to calculate the indicators. If None, the
        end of the data is used. Default is None.
    risk_free : float, optional
        The risk-free rate to use in the calculations. Default is 0.

    Returns
    -------
    pandas.DataFrame
        A DataFrame with the calculated indicators.
    """
    indicators_names, indicators_values = [], []
    for indicator in __available_indicators__:
        indicators_names.append(indicator)
        indicators_values.append(getattr(self, indicator)(start, end, risk_free))

    return pd.DataFrame(indicators_values, index=indicators_names, columns=["value"])


@assert_ts
def cumulative(self, start=None, end=None):
    """
    Calculate the cumulative return for the given Table or TableSeries.

    Cumulative return is the total change in price of an investment over a set time period.
    It includes the compounding of returns.

    Parameters
    ----------
    start : str or datetime-like, optional
        Start date for the period to calculate the cumulative return. If None, the
        start of the data is used. Default is None.
    end : str or datetime-like, optional
        End date for the period to calculate the cumulative return. If None, the
        end of the data is used. Default is None.

    Returns
    -------
    pandas.Series or pandas.DataFrame
        The cumulative return.
    """

    return (self.loc[start:end] + 1).cumprod() - 1
