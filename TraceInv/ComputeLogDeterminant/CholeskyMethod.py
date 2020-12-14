# =======
# Imports
# =======

import numpy
import scipy
from scipy import linalg

try:
    import sksparse
    from sksparse.cholmod import cholesky
    SuiteSparseInstalled = True
except:
    SuiteSparseInstalled = False
SuiteSparseInstalled = False

from .._LinearAlgebra import SparseCholesky

# ===============
# Cholesky Method
# ===============

def CholeskyMethod(A):
    """
    Computes log-determinant using Cholesky decomposition.

    This function is essentially a wrapper for the Choleksy function of the scipy and scikit-sparse packages,
    and primarily used for testing and comparison (benchmarking)  against the randomized methods that are
    implemented in this package.

    :param A: A full-rank matrix.
    :type A: numpy.ndarray

    The log-determinant is computed from the Cholesky decomposition :math:`\mathbf{A} = \mathbf{L} \mathbf{L}^{\intercal}` as

    .. math::

        \log | \mathbf{A} | = 2 \mathrm{trace}( \log \mathrm{diag}(\mathbf{L})).

    .. note::
        This function uses the `Suite Sparse <https://people.engr.tamu.edu/davis/suitesparse.html>`_ 
        package to compute the Cholesky decomposition. See the :ref:`installation <InstallScikitSparse>`.
    
    The result is exact (no approximation) and should be used as benchmark to test other methods.
    """

    if scipy.sparse.issparse(A):

        if SuiteSparseInstalled:
            # Use Suite Sparse
            Factor = sksparse.cholmod.cholesky(A)
            LogDet = Factor.logdet()
        else:
            # Use scipy
            L_diag = SparseCholesky(A,DiagonalOnly=True)
            LogDet = 2.0 * numpy.sum(numpy.log(L_diag))

    else:

        # Use scipy
        L = scipy.linalg.cholesky(A,lower=True)
        LogDet = 2.0 * numpy.sum(numpy.log(numpy.diag(L)))

    return LogDet
