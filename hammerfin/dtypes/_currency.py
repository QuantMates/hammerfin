import numpy as np

from ._custom_dtype import CustomDType


class Currency(CustomDType):
    """Custom dtype for currency values"""

    def __init__(self, currency="USD", inflation_adjusted=False, base_inflation_date=None):
        self.currency = currency
        self.inflation_adjusted = inflation_adjusted
        self.base_inflation_date = base_inflation_date
        super().__init__(np.float64)

    def __repr__(self):
        return f"Currency('{self.currency}', {self.inflation_adjusted}, {self.base_inflation_date})"

    def __str__(self):
        return f"Currency('{self.currency}', {self.inflation_adjusted}, {self.base_inflation_date})"

    def __eq__(self, other):
        if isinstance(other, Currency):
            return (
                self.currency == other.currency
                and self.inflation_adjusted == other.inflation_adjusted
                and self.base_inflation_date == other.base_inflation_date
            )
        return False
