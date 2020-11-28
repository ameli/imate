# =======
# Imports
# =======

import numpy
import scipy
from scipy import sparse
from scipy import linalg
from scipy.sparse import linalg

# =============
# Linear Solver
# =============

def LinearSolver(A,b,Tol=1e-5):
    """
    Solves the linear system :math:`Ax = b` where :math:`A` can be either sparse or dense.

    :param A: matrix of coefficients, two-dimensional array, can be either sparse or dense
    :type A: numpy.ndarray

    :param b: column vector of the right hand side of the linear system, one-dimensional array
    :type b: array

    :param Tol: Tolerance for the error of solving linear system. This is only applicable if ``A`` is sparse.
    :type Tol: float

    :return: one-dimensional array of the solution of the linear system
    :rtype: numpy.array
    """

    if scipy.sparse.isspmatrix(A):

        # Use direct method
        # x = scipy.sparse.linalg.spsolve(A,b)

        # Use iterative method
        if b.ndim == 1:
            x = scipy.sparse.linalg.cg(A,b,tol=Tol)[0]
        else:
            x = numpy.zeros(b.shape)
            for i in range(x.shape[1]):
                x[:,i] = scipy.sparse.linalg.cg(A,b[:,i],tol=Tol)[0]
    else:
        # Dense matrix
        x = scipy.linalg.solve(A,b,sym_pos=True)

    return x
