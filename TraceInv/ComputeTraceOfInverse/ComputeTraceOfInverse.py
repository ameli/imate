# =======
# Imports
# =======

from .CholeskyMethod import CholeskyMethod
from .HutchinsonMethod import HutchinsonMethod
from .StochasticLanczosQuadratureMethod import StochasticLanczosQuadratureMethod

# ========================
# Compute Trace Of Inverse
# ========================

def ComputeTraceOfInverse(A,Method='cholesky',**Options):
    """
    Computes the trace of inverse of a matrix without using interpolation.

    The trace of inverse is computed with one of these three methods in this function.

    =============================  =============
    Method                             Type
    =============================  =============
    Cholesky                       exact
    Hutchinson                     approximation
    Stochastic Lanczos Quadrature  approximation
    =============================  =============

    :param A: Invertible matrix
    :type A: ndarray

    :param Method: One of ``'cholesky'``, ``'hutchinson'``, or ``'SLQ'``. Default if ``'cholesky'``.
    :type Method: string

    :param Options: Options for either of the methods. 
    :type Options: ``'**kwargs'``

    :return: trace of inverse of matrix
    :rtype: float

    :raises RunTimeError: Method is not recognized.
    """

    if Method == 'cholesky':
        Trace = CholeskyMethod(A,**Options)

    elif Method == 'hutchinson':
        Trace = HutchinsonMethod(A,**Options)

    elif Method == 'SLQ':
        Trace = StochasticLanczosQuadratureMethod(A,**Options)

    else:
        raise RuntimeError('Method: %s is not recognized.'%Method)

    return Trace
