from __future__ import annotations

from typing import Sequence

import numpy as np


def object_array(values: Sequence[object]) -> np.ndarray:
    result = np.empty(len(values), dtype=object)
    for index, value in enumerate(values):
        result[index] = value
    return result
