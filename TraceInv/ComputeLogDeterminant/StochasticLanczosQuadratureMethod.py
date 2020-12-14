# =======
# Imports
# =======

import numpy
from numpy import linalg

from .._LinearAlgebra import LanczosTridiagonalization
from .._LinearAlgebra import GolubKahnBidiagonalization

# ====================================
# Stochastic Lanczos Quadrature Method
# ====================================

def StochasticLanczosQuadratureMethod(
        A,
        NumIterations=20,
        LanczosDegree=20,
        UseLanczosTridiagonalization=False):
    """
    Computes the log-determinant of a matrix based on stochastic Lanczos quadrature method.
    
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

        In Lanczos tri-diagonalization method, ``theta`` is the eigenvalues of ``T``. 
        However, in Golub-Kahn bi-diagonalization method, ``theta`` is the singular values of ``B``.
        The relation between these two methods are are follows: ``B.T*B`` is the ``T`` for ``A.T*A``.
        That is, if we have the input matrix ``A.T*T``, its Lanczos tri-diagonalization ``T`` is the 
        same matrix as if we bi-diagonalize ``A`` (not ``A.T*A``) with Golub-Kahn to get ``B``, 
        then ``T = B.T*B``. This has not been highlighted paper referenced below.

        To correctly implement Golub-Kahn, here Theta should be the singular values of ``B``, *NOT*
        the square of the singular values of ``B`` (as described in the paper incorrectly!).
        
    Reference:
        * `Ubaru, S., Chen, J., and Saad, Y. (2017) <https://www-users.cs.umn.edu/~saad/PDF/ys-2016-04.pdf>`_, 
          Fast Estimation of :math:`\\mathrm{tr}(F(A))` Via Stochastic Lanczos Quadrature, SIAM J. Matrix Anal. Appl., 38(4), 1075-1099.
    """

    n = A.shape[0]
    LogDetEstimates = numpy.zeros((NumIterations,))

    for i in range(NumIterations):

        # Radamacher random vector, consists of 1 and -1.
        w = numpy.sign(numpy.random.randn(n))

        if UseLanczosTridiagonalization:
            # Lanczos recursive iteration to convert A to tri-diagonal form T
            T = LanczosTridiagonalization(A,w,LanczosDegree,Tolerance=1e-10)

            # Spectral decomposition of T
            Eigenvalues,Eigenvectors = numpy.linalg.eigh(T)

            Theta = numpy.abs(Eigenvalues)
            Tau2 = Eigenvectors[0,:]**2

        else:

            # Use Golub-Kahn-Lanczos bi-diagonalization instead of Lanczos tri-diagonalization
            B = GolubKahnBidiagonalization(A,w,LanczosDegree,Tolerance=1e-10)
            LeftEigenvectors,SingularValues,RightEigenvectorsTransposed = numpy.linalg.svd(B)
            Theta = SingularValues   # Theta is just singular values, not singular values squared
            Tau2 = RightEigenvectorsTransposed[:,0]**2

        # Here, f(theta) = log(theta), since log det X = trace of log X.
        LogDetEstimates[i] = numpy.sum(Tau2 * (numpy.log(Theta))) * n

    LogDet = numpy.mean(LogDetEstimates)

    return LogDet
