from dataclasses import dataclass

import numpy as np


@dataclass
class FinDType:
    """Base class for HammerFin dtypes"""

    numpy_dtype: np.dtype = None


def assert_fin_dtype(func):
    """Assert that the TableSeries has a fin_dtype"""

    def wrapper(*args, **kwargs):
        fin_dtype = args[0].fin_dtype
        if not fin_dtype:
            raise TypeError("TableSeries must have a fin_dtype")
        return func(*args, **kwargs)

    return wrapper
