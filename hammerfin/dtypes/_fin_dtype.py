from dataclasses import dataclass

import numpy as np


@dataclass
class FinDType:
    """Base class for HammerFin dtypes"""

    numpy_dtype: np.dtype = None
