# =======
# Imports
# =======

from .CholeskyMethod import CholeskyMethod
from .HutchinsonMethod import HutchinsonMethod
# from .StochasticLanczosQuadratureMethod import StochasticLanczosQuadratureMethod
from .StochasticLanczosQuadratureMethod_Parallel import StochasticLanczosQuadratureMethod
# from TraceInv.ComputeTraceOfInverse.StochasticLanczosQuadratureMethod_Parallel import StochasticLanczosQuadratureMethod

# ========================
# Compute Trace Of Inverse
# ========================

def ComputeTraceOfInverse(A,ComputeMethod='cholesky',**Options):
    """
    Computes the trace of inverse of a matrix. 
    See :ref:`Compute Trace of Inverse User Guide<ComputeTraceOfInverse_UserGuide>` for details.

    :param A: Invertible matrix
    :type A: ndarray

    :param ComputeMethod: One of ``'cholesky'``, ``'hutchinson'``, or ``'SLQ'``. Default if ``'cholesky'``.
    :type ComputeMethod: string

    :param Options: Options for either of the methods. 
    :type Options: ``**kwargs``

    :return: trace of inverse of matrix
    :rtype: float

    :raises RunTimeError: Method is not recognized.

    **Methods:**

    The trace of inverse is computed with one of these three methods in this function.

    ===================  =============================  =============
    ``'ComputeMethod'``  Description                    Type
    ===================  =============================  =============
    ``'cholesky'``       Cholesky method                exact
    ``'hutchinson'``     Hutchinson method              approximation
    ``'SLQ'``            Stochastic Lanczos Quadrature  approximation
    ===================  =============================  =============

    Depending the method, this function calls these modules:

    * :mod:`TraceInv.ComputeTraceOfInverse.CholeskyMethod`
    * :mod:`TraceInv.ComputeTraceOfInverse.HutchinsonMethod`
    * :mod:`TraceInv.ComputeTraceOfInverse.StochasticLanczosQuadratureMethod`

    **Examples:**

    .. code-block:: python

       >>> from TraceInv import GenerateMatrix
       >>> from TraceInv import ComputeTraceOfInverse
       
       >>> # Generate a symmetric positive-definite matrix of the shape (20**2,20**2)
       >>> A = GenerateMatrix(NumPoints=20)
       
       >>> # Compute trace of inverse
       >>> trace = ComputeTraceOfInverse(A)

    The above example uses the Cholesky method by default.
    In the next example, we apply the *Hutchinson's randomized estimator* method.

    .. code-block:: python

       >>> trace = ComputeTraceOfInverse(A,ComputeMethod='hutchinson',NumIterations=20)

    Using the stochastic Lanczos quadrature method with Lanczos tri-diagonalization

    .. code-block:: python

       >>> trace = ComputeTraceOfInverse(A,ComputeMethod='SLQ',NumIterations=20,LanczosDegree=30)

    Using the stochastic Lanczos quadrature method with Golub-Kahn bi-diagonalization

    .. code-block:: python

       >>> trace = ComputeTraceOfInverse(A,ComputeMethod='SLQ',NumIterations=20,LanczosDegree=30,UseLanczosTridiagonalization=False)
    """

    if ComputeMethod == 'cholesky':
        Trace = CholeskyMethod(A,**Options)

    elif ComputeMethod == 'hutchinson':
        Trace = HutchinsonMethod(A,**Options)

    elif ComputeMethod == 'SLQ':
        Trace = StochasticLanczosQuadratureMethod(A,**Options)

    else:
        raise RuntimeError('Method: %s is not recognized.'%ComputeMethod)

    return Trace
