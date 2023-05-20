import numpy as np
from pandas.api.extensions import ExtensionArray, ExtensionDtype

# pylint: disable = missing-function-docstring


class CurrencyDtype(ExtensionDtype):
    """
    A custom pandas dtype for currency data.

    Attributes
    ----------
    type : type
        The type of the data, which is float.
    kind : str
        The kind of the data, which is "f" for float.
    name : str
        The name of the dtype, which is "currency".
    na_value : float
        The value used to represent missing data, which is np.nan.
    """

    type = float
    kind = "f"
    name = "currency"
    na_value = np.nan

    def __init__(self, currency="USD", inflation_adjusted=False, base_year=None):
        """
        Initialize the CurrencyDtype.

        Parameters
        ----------
        currency : str, optional
            The currency of the data. Default is "USD".
        inflation_adjusted : bool, optional
            Whether the data is adjusted for inflation. Default is False.
        base_year : int, optional
            The base year for inflation adjustment. Default is None.
        """

        self.currency = currency
        self.inflation_adjusted = inflation_adjusted
        self.base_year = base_year

    @classmethod
    def construct_array_type(cls):
        return CurrencyArray


class CurrencyArray(ExtensionArray):
    """
    A custom pandas ExtensionArray for currency data.

    Attributes
    ----------
    _data : numpy.ndarray
        The underlying data.
    _dtype : CurrencyDtype
        The dtype of the data.
    """

    def __init__(self, values, dtype):
        self._data = np.asarray(values, dtype="float")
        self._dtype = dtype

    @property
    def dtype(self):
        return self._dtype

    def __getitem__(self, item):
        return self._data[item]

    def __len__(self):
        return len(self._data)

    def __array__(self, dtype=None):  # pylint: disable = unused-argument
        return self._data

    def isna(self):
        return np.isnan(self._data)

    @classmethod
    def _from_sequence(cls, scalars, dtype=None, copy=False):  # pylint: disable = unused-argument
        return cls(scalars, dtype)

    @classmethod
    def _from_factorized(cls, values, original):
        return cls(values, original.dtype)

    def copy(self):
        return CurrencyArray(self._data.copy(), self._dtype)

    def _reduce(self, name, skipna=True, ddof=0):
        """
        Perform a reduction operation.

        This should return a single value (like mean, sum, min, etc.)
        """
        if name == "sum":
            return np.nansum(self._data) if skipna else np.sum(self._data)
        if name == "mean":
            return np.nanmean(self._data) if skipna else np.mean(self._data)
        if name == "min":
            return np.nanmin(self._data) if skipna else np.min(self._data)
        if name == "max":
            return np.nanmax(self._data) if skipna else np.max(self._data)
        if name == "var":
            return np.nanvar(self._data, ddof=ddof) if skipna else np.var(self._data, ddof=ddof)
        if name == "std":
            return np.nanstd(self._data, ddof=ddof) if skipna else np.std(self._data, ddof=ddof)
        raise TypeError(
            f"Cannot perform reduction '{name}' with skipna={skipna} and ddof={ddof} on {self.__class__.__name__}"
        )

    def fillna(self, value, method=None, limit=None):
        if method is None:
            self._data[self.isna()] = value
        elif method == "pad":
            # Forward fill missing values
            mask = self.isna()
            forward_fill_values = np.concatenate(([value], self._data[:-1]))  # Shift data forward by one element
            if limit is not None:
                # Apply limit to consecutive missing values
                consecutive_missing = np.cumsum(mask)
                forward_fill_values[consecutive_missing > limit] = np.nan
            self._data[mask] = forward_fill_values[mask]
        elif method == "backfill":
            # Backward fill missing values
            mask = self.isna()
            backward_fill_values = np.concatenate((self._data[1:], [value]))  # Shift data backward by one element
            if limit is not None:
                # Apply limit to consecutive missing values
                consecutive_missing = np.cumsum(mask[::-1])[::-1]
                backward_fill_values[consecutive_missing > limit] = np.nan
            self._data[mask] = backward_fill_values[mask]
        else:
            raise ValueError(f"Invalid fill method '{method}'. Supported methods are: None, 'pad', 'backfill'")
        return self

    def __add__(self, other):
        return self.__class__(self._data + other._data, self._dtype)

    def __sub__(self, other):
        return self.__class__(self._data - other._data, self._dtype)

    def __mul__(self, other):
        return self.__class__(self._data * other._data, self._dtype)

    def __div__(self, other):
        return self.__class__(self._data / other._data, self._dtype)

    def __eq__(self, other):
        return self._data == other._data

    def __ne__(self, other):
        return self._data != other._data

    def __lt__(self, other):
        return self._data < other._data

    def __le__(self, other):
        return self._data <= other._data

    def __gt__(self, other):
        return self._data > other._data

    def __ge__(self, other):
        return self._data >= other._data

    def take(self, indices, allow_fill=False, fill_value=None):
        if allow_fill:
            fill_value = self.dtype.na_value if fill_value is None else fill_value
            mask = indices == -1
            if (indices < -1).any():
                raise ValueError("Invalid value in 'indices'.")
            out = np.empty(indices.shape, dtype=self.dtype)
            out[mask] = fill_value
            out[~mask] = self._data[indices[~mask]]
            return self.__class__(out, self._dtype)
        return self.__class__(self._data.take(indices), self._dtype)

    def change_currencies(self, new_currency):
        """
        TODO: Implement this method

        --------
        exchange_rate = get_exchange_rate(self._dtype.currency, new_currency)

        self._data *= exchange_rate
        self._dtype = CurrencyDtype(new_currency)
        """
        self._dtype = CurrencyDtype(
            currency=new_currency,
            inflation_adjusted=self._dtype.inflation_adjusted,
            base_year=self._dtype.base_year,
        )
        return self

    def adjust_for_inflation(self, base_year):  # pylint: disable = unused-argument
        """
        TODO: Implement this method
        """
        return self
