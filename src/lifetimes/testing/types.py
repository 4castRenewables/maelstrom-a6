import datetime
from typing import Union

import numpy as np

Timestamp = Union[str, datetime.datetime]
CoordinateDict = dict[str, np.ndarray]
Coordinate = tuple[float, float]
