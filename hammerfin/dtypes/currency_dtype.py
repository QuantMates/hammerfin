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
    """

    type = float
    kind = "f"
    name = "currency"

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
