# =======
# Imports
# =======

import numpy
import scipy
from scipy import linalg

# Package
from .LinearSolver import LinearSolver

# =================
# Hutchinson Method
# =================

def HutchinsonMethod(A,NumIterations=20):
    """
    Computes the trace of inverse of a matrix by Hutchinson method.

    The random vectors have Radamacher distribution. Compared to the Gaissuan
    distribution, the former distribution yields estimation of trace with lower
    variance.

    :param A: invertible matrix
    :type A: ndarray

    :param NumIterations: number of Monte-Carlo random trials
    :type NumIterations: int

    :return: Trace of matrix ``A``.
    :rtype: float
    """

    n = A.shape[0]

    # Create a random matrix with m random vectors with Radamacher distribution.
    E = numpy.sign(numpy.random.randn(n,NumIterations))

    # Orthonormalize random vectors
    Q,R = scipy.linalg.qr(E,mode='economic',overwrite_a=True,pivoting=False,check_finite=False)
    AinvQ = LinearSolver(A,Q)
    QtAinvQ = numpy.matmul(Q.T,AinvQ)

    # Trace
    Trace = n*numpy.mean(numpy.diag(QtAinvQ))

    return Trace
