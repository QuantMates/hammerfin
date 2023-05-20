from dataclasses import dataclass

import numpy as np


@dataclass
class CustomDType:
    """Base class for custom dtypes"""

    numpy_dtype: np.dtype = None
