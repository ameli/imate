# =======
# Imports
# =======

import numpy
import scipy
from scipy import sparse
from scipy.sparse import linalg

# ===============
# Sparse Cholesky
# ===============

def SparseCholesky(A,DiagonalOnly=False):
    """
    This function uses LU decomposition assuming that ``A`` is symmetric and positive-definite.

    .. note::

        This function does not check if ``A`` is positive-definite. If the input matrix is 
        not positive-definite, the Cholesky decomposition does not exist and the return value
        is misleadingly wrong.

    :param A: Symmetric and positive-definite matrix.
    :type A: ndarray

    :param DiagonalOnly: If ``True``, returns a column array of the diagonals of the Cholesky decomposition.
        If ``False``, returns the full Cholesky matrix as scipy.sparse.csc_matrix.

    :return: Chlesky decomposition of ``A``.
    :rtype: Super.LU
    """

    n = A.shape[0]

    # sparse LU decomposition
    LU = scipy.sparse.linalg.splu(A,diag_pivot_thresh=0,permc_spec='NATURAL')

    if DiagonalOnly:

        # Return diagonals only
        return numpy.sqrt(LU.U.diagonal())

    else:

        # return LU.L.dot(sparse.diags(LU.U.diagonal()**0.5))

        # check the matrix A is positive definite.
        if (LU.perm_r == numpy.arange(n)).all() and (LU.U.diagonal() > 0).all():
            return LU.L.dot(sparse.diags(LU.U.diagonal()**0.5))
        else:
            raise RuntimeError('The matrix is not positive-definite.')
