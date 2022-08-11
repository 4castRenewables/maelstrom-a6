import datetime
import typing as t

import numpy as np

Timestamp = t.Union[str, datetime.datetime]
CoordinateDict = dict[str, np.ndarray]
Coordinate = tuple[float, float]
