"""Utility functions for Freckll."""

import numpy as np
import numpy.typing as npt
from scipy import sparse

def n_largest_index(array: npt.NDArray, n: int, axis: int = 0) -> npt.NDArray[np.integer]:
    """Return the indices of the n largest elements along the given axis."""
    return np.argsort(array, axis=axis)[-1 : -n - 1 : -1]


def convert_to_banded(mat: sparse.sparray, band: int) -> npt.NDArray[np.float64]:
    import numpy as np
    from scipy.sparse import find
    lower_band = band#mat[:, 0].indices[-1] or mat[0].indices[-1] 
    #lower_band = find(mat[:,0])[0][-1]

    upper_band = lower_band


    ab = np.zeros(shape=(upper_band+lower_band+1,mat.shape[0]))
    # diag_index = np.arange(0,mat.shape[0])
    # diagonals = [(kth_diag_indices(mat,x),mat.diagonal(x)) for x in range(-lower_band,upper_band)]
    row, col, values = find(mat)
    ab[upper_band + row - col, col] = values
    # for indices,vals in diagonals:
    #     row,col = indices
    #     ab[upper_band+row-col,col] = mat[row,col]


    return ab

def convert_to_banded_lsoda(mat: sparse.sparray,band: int):
    import numpy as np
    from scipy.sparse import find
    lower_band = band
    upper_band = lower_band
    
    ab = np.zeros(shape=(upper_band+lower_band*2+1,mat.shape[0]))
    row, col, values = find(mat)
    ab[upper_band + row - col, col] = values



    return ab