from typing import Generic
import numpy.typing as npt
import numpy as np

from pint import Quantity

ComplexArray = npt.NDArray[np.complexfloating]
FloatArray = npt.NDArray[np.floating]
IntArray = npt.NDArray[np.integer]
StrArray = npt.NDArray[np.str_]
