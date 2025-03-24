"""Standard types for FRECKLL."""

import typing as t

import numpy as np
import numpy.typing as npt

FreckllArray = npt.NDArray[np.floating]
FreckllArrayInt = npt.NDArray[np.integer]

ReactionFunction = t.Callable[
    [FreckllArray],
    list[FreckllArray],
]
