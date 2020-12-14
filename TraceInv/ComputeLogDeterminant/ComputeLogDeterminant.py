# =======
# Imports
# =======

from .CholeskyMethod import CholeskyMethod
from .StochasticLanczosQuadratureMethod import StochasticLanczosQuadratureMethod
# from .StochasticLanczosQuadratureMethod_Parallel import StochasticLanczosQuadratureMethod

# __all__ = ['ComputeLogDeterminant']

# =======================
# Compute Log Determinant
# =======================

def ComputeLogDeterminant(A,ComputeMethod='cholesky',NumIterations=20,LanczosDegree=20,UseLanczosTridiagonalization=False):
    """
    Computes the log-determinant of full-rank matrix ``A``.

    :param A: A full-rank matrix.
    :type A: numpy.ndarray or scipy.sparse.csc_matrix

    :param NumIterations: Number of Monte-Carlo trials
    :type NumIterations: int

    :param LanczosDegree: Lanczos degree of the tri-diagonalization (or bi-diaqgonalization) process
    :type LanczosDegree: int

    :param UseLanczosTridiagonalization: Flag, if ``True``, it uses the Lanczos tri-diagonalization. 
        If ``False``, it uses the Golub-Kahn bi-diagonalization.
    :type UseLanczosTridiagonalization: bool

    :return: Log-determinant of ``A``
    :rtype: float

    .. note::

        For computing the *trace of inverse* with the stochastic Lanczos quadrature method 
        (see :mod:`TraceInv.ComputeTraceOfInverse.StochasticLanczosQuadrature`), the preferred algorithm is
        the Lanczos tri-diagonalization, as opposed to Golub-Kahn bi-diagonalization.

        In contrast to the above, the preferred SLQ method for computing *log-determinant* is
        the Golub-Kahn bi-diagonalization. The reason is that if the matrix :math:`\mathbf{A}` 
        has many singular values close to zero, bi-diagonalization performs better, 
        and this matters when we compute determinant.
    
    References:
        * Ubaru, S., Chen, J., and Saad, Y. (2017). 
          Fast estimation of :math:`\mathrm{tr}(f(A))` via stochastic Lanczos quadrature. 
          *SIAM Journal on Matrix Analysis and Applications*, 38(4), 1075-1099. 
          `doi: 10.1137/16M1104974 <https://doi.org/10.1137/16M1104974>`_

    **Examples:**

    .. code-block:: python

        >>> # Import packages
        >>> from TraceInv import ComputeLogDeterminant
        >>> from TraceInv import GenerateMatrix

        >>> # Generate a sample matrix
        >>> A = GenerateMatrix(NumPoints=20)

        >>> # Compute log-determinant with Cholesky method
        >>> LogDet_1 = ComputeLogDeterminant(A)

        >>> # Compute log-determinant with stochastic Lanczos quadrature method
        >>> LogDet_1 = ComputeLogDeterminant(A,ComputeMethod='SLQ',NumIterations=20,LanczosDegree=20)

        >>> # Compute log-determinant with stochastic Lanczos quadrature method with Golub-Khan bi-diagonalization
        >>> LogDet_1 = ComputeLogDeterminant(A,ComputeMethod='SLQ',NumIterations=20, 
        ...                 LanczosDegree=20,Tridiagonalization=False)
    """

    if ComputeMethod == 'cholesky':

        # Cholesky method
        LogDet = CholeskyMethod(A)

    elif ComputeMethod == 'SLQ':

        # Stochastic Lanczos Quadrature method (by Monte-Carlo sampling)
        LogDet = StochasticLanczosQuadratureMethod(A,NumIterations,LanczosDegree,UseLanczosTridiagonalization)

    else:
        raise ValueError('ComputeMethod is invalid.')

    return LogDet
