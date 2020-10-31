# =======
# Imports
# =======

import numpy
from numpy import linalg

from .LinearAlgebra import LanczosTridiagonalization
from .LinearAlgebra import LanczosTridiagonalization2
from .LinearAlgebra import GolubKahnLanczosBidiagonalization

# ====================================
# Stochastic Lanczos Quadrature Method
# ====================================

def StochasticLanczosQuadratureMethod(A,
        NumIterations=20,
        LanczosDegree=20,
        UseLanczosTridiagonalization=False):
    """
    Computes the trace of inverse of matrix based on stochastic Lanczos quadrature method.
   
    Reference
        * Ubaru, S., Chen, J., and Saad, Y. (2017), `Fast Estimation of :math:`\mathrm{tr}(F(A))` Via Stochastic Lanczos Quadrature <https://www-users.cs.umn.edu/~saad/PDF/ys-2016-04.pdf>`_, SIAM J. Matrix Anal. Appl., 38(4), 1075-1099.

    .. note::

        In Lanczos tridiagonalization method, :math:`\\theta`` is the eigenvalue of ``T``. 
        However, in Golub-Kahn bidoagonalization method, :math:`\\theta` is the singular values of ``B``.
        The relation between these two methods are are follows: ``B.T*B`` is the ``T`` for ``A.T*A``.
        That is, if we have the input matrix ``A.T*T``, its Lanczos tridiagonalization ``T`` is the same matrix
        as if we bidiagonalize ``A`` (not ``A.T*A``) with Golub-Kahn to get ``B``, then ``T = B.T*B``.
        This has not been highlighted paper in the above paper.

        To correctly implement Golub-Kahn, here :math:`\\theta` should be the singular values of ``B``, **NOT**
        the square of the singular values of ``B`` (as decribed in the above paper incorrectly!).


    :param A: invertible matrix
    :type A: ndarray

    :param NumIterations: Number of Monte-Carlo trials
    :type NumIterations: int

    :param LanczosDegree: Lanczos degree
    :type LanczosDegree: int

    :param UseLanczosTridiagonalization: Flag, if ``True``, it uses the Lanczos tridiagonalization. 
        If ``False``, it uses the Golub-Kahn bi-diagonalization.
    :type UseLanczosTridiagonalization: bool

    :return: Trace of ``A``
    :rtype: float
    """

    n = A.shape[0]
    TraceEstimates = numpy.zeros((NumIterations,))

    for i in range(NumIterations):

        # Radamacher random vector, consists of 1 and -1.
        w = numpy.sign(numpy.random.randn(n))

        if UseLanczosTridiagonalization:
            # Lanczos recustive iteration to convert A to tridiagonal form T
            # T = LanczosTridiagonalization(A,w,LanczosDegree)
            T = LanczosTridiagonalization2(A,w,LanczosDegree)

            # Spectral decomposition of T
            Eigenvalues,Eigenvectors = numpy.linalg.eigh(T)

            Theta = numpy.abs(Eigenvalues)
            Tau2 = Eigenvectors[0,:]**2

        else:

            # Use Golub-Kahn-Lanczos bidigonalization instead of Lanczos tridiagonalization
            B = GolubKahnLanczosBidiagonalization(A,w,LanczosDegree)
            LeftEigenvectors,SingularValues,RightEigenvectorsTransposed = numpy.linalg.svd(B)
            Theta = SingularValues    # Theta is just singular values, not singular values squared
            Tau2 = RightEigenvectorsTransposed[:,0]**2

        # Here, f(theta) = 1/theta, since we compute trace of matrix inverse
        TraceEstimates[i] = numpy.sum(Tau2 * (1.0/Theta)) * n

    Trace = numpy.mean(TraceEstimates)

    return Trace
