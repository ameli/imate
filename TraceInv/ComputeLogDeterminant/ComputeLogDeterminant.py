# =======
# Imports
# =======

import numpy
import scipy
from scipy import linalg
# import multiprocessing

try:
    import sksparse
    from sksparse.cholmod import cholesky
    SuiteSparseInstalled = True
except:
    SuiteSparseInstalled = False
SuiteSparseInstalled = False

from ..LinearAlgebra import LinearSolver
from ..LinearAlgebra import SparseCholesky
from ..LinearAlgebra import LanczosTridiagonalization
from ..LinearAlgebra import LanczosTridiagonalization2
from ..LinearAlgebra import GolubKahnLanczosBidiagonalization

# __all__ = ['ComputeLogDeterminant']

# ===============
# Cholesky Method
# ===============

def CholeskyMethod(A):
    """
    Computes log-determinant using Cholesky decomposition.

    :param A: A full-rank matrix.
    :type A: numpy.ndarray

    The log-determinant is computed from the Cholesky decomposition :math:`\mathbf{A} = \mathbf{L} \mathbf{L}^{\intercal}` as

    .. math::

        \log | \mathbf{A} | = 2 \mathrm{trace}( \log \mathrm{diag}(\mathbf{L})).

    .. note::
        This function uses the `Suite Sparse <https://people.engr.tamu.edu/davis/suitesparse.html>`_ 
        package to compute the Cholesky decompositon. See the :ref:`installation <InstallScikitSparse>`.
    
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

# ====================================
# Stochastic Lanczos Quadrature Method
# ====================================

def StochasticLanczosQuadratureMethod(A,LanczosDegree=20,UseLanczosTridiagonalization=False):
    """
    Method should be 'LanczosTridiagonalization' or 'GolubKahnLanczosBidiagonalization'.
    LanczosDegree is an integer.

    :param A: A full-rank matrix.
    :type A: numpy.ndarray or scipy.sparse.csc_matrix

    :param NumIterations: Number of Monte-Carlo trials
    :type NumIterations: int

    :param LanczosDegree: Lanczos degree of the tri-diagonalization (or bi-diaqgonalization) process
    :type LanczosDegree: int

    :param UseLanczosTridiagonalization: Flag, if ``True``, it uses the Lanczos tridiagonalization. 
        If ``False``, it uses the Golub-Kahn bi-diagonalization.
    :type UseLanczosTridiagonalization: bool

    :return: Log-determinant of ``A``
    :rtype: float

    About the implementation of Golub-Kahn bi-diagonalization:
        In Lanczos tridiagonalization method, ``theta`` is the eigenvalues of ``T``. 
        However, in Golub-Kahn bidoagonalization method, ``theta`` is the singular values of ``B``.
        The relation between these two methords are are follows: ``B.T*B`` is the ``T`` for ``A.T*A``.
        That is, if we have the input matrix ``A.T*T``, its Lanczos tridiagonalization ``T`` is the same matrix
        as if we bidiagonalize ``A`` (not ``A.T*A``) with Golub-Kahn to get ``B``, then ``T = B.T*B``.
        This has not been highlighted paper referenced below.

        To correctly implement Golub-Kahn, here Theta should be the singular values of ``B``, *NOT*
        the square of the singular values of ``B`` (as decribed in the paper incorrectly!).
    """

    # Radamacher random vector, consists of 1 and -1.
    n = A.shape[0]
    w = numpy.sign(numpy.random.randn(n))

    if UseLanczosTridiagonalization:
        # Lanczos recustive iteration to convert A to tridiagonal form T
        # T = LanczosTridiagonalization(A,w,LanczosDegree)
        T = LanczosTridiagonalization2(A,w,LanczosDegree,Tolerance=1e-10)

        # Spectral decomposition of T
        Eigenvalues,Eigenvectors = numpy.linalg.eigh(T)

        Theta = numpy.abs(Eigenvalues)
        Tau2 = Eigenvectors[0,:]**2

        # Here, f(theta) = log(theta), since log det is trace of log
        LogDetEstimate = numpy.sum(Tau2 * (numpy.log(Theta))) * n

    else:

        # Use Golub-Kahn-Lanczos bidigonalization instead of Lanczos tridiagonalization
        B = GolubKahnLanczosBidiagonalization(A,w,LanczosDegree,Tolerance=1e-10)
        LeftEigenvectors,SingularValues,RightEigenvectorsTransposed = numpy.linalg.svd(B)
        Theta = SingularValues   # Theta is just singular values, not singular values squared
        Tau2 = RightEigenvectorsTransposed[:,0]**2

        # Here, f(theta) = log(theta), since log det X = trace of log X.
        LogDetEstimate = numpy.sum(Tau2 * (numpy.log(Theta))) * n

    return LogDetEstimate

# ====================
# Monte Carlo Sampling
# ====================

def MonteCarloSampling(A,NumIterations=20,LanczosDegree=20,UseLanczosTridiagonalization=False):
    """
    This function samples results from the Stochastic Lanczos Quadrature method.

    :param A: A full-rank matrix.
    :type A: numpy.ndarray or scipy.sparse.csc_matrix

    :param NumIterations: Number of Monte-Carlo trials
    :type NumIterations: int

    :param LanczosDegree: Lanczos degree of the tri-diagonalization (or bi-diaqgonalization) process
    :type LanczosDegree: int

    :param UseLanczosTridiagonalization: Flag, if ``True``, it uses the Lanczos tridiagonalization. 
        If ``False``, it uses the Golub-Kahn bi-diagonalization.
    :type UseLanczosTridiagonalization: bool

    :return: Log-determinant of ``A``
    :rtype: float

    .. note::

        The parallelized implementation requires the ``ray`` package (see :ref:`installation <InstallRay>`).
        Currently, the parallel implementation is commented in the code below.
    """

    # No Parallel processing
    LogDetEstimatesList = [StochasticLanczosQuadratureMethod(A,LanczosDegree,UseLanczosTridiagonalization) for i in range(NumIterations)]

    # Parallel processing with Ray
    # Get number of cpus
    # NumProcessors = psutil.cpu_count()

    # Parallelization with ray
    # ray.init(num_cpus=NumProcessors,logging_level=logging.FATAL)

    # Put A into object store
    # A_id = ray.put(A)

    # Parallel section with ray. This just creates process Ids. It does not do computation
    # Process_Ids = [MonteCarloSampling.remote(A_id,Method,LanczosDegree) for i in range(NumIterations)]

    # Do the parallel computations
    # LogDetEstimatesList = ray.get(Process_Ids)

    # ray.shutdown()

    # LogDet = numpy.mean(numpy.array(LogDetEstimatesList))

    # Parallel processing with multiprocessing
    # NumProcessors = multiprocessing.cpu_count()
    # pool = multiprocessing.Pool(processes=NumProcessors)
    # ChunkSize = int(NumIterations / NumProcessors)
    # if ChunkSize < 1:
    #     ChunkSize = 1

    # MonteCarloSampling_PartialFunction = partial(LikelihoodEstimation.MonteCarloSampling,A,Method,LanczosDegree)

    # Iterations = range(NumIterations)
    # Processes = [pool.apply_async(LikelihoodEstimation.MonteCarloSampling,(A,Method,LanczosDegree)) for i in range(NumIterations)]
    # pool.close()
    # pool.join()
    # LogDetEstimatesList = [Process.get() for Process in Processes]

    # Average samples
    LogDet = numpy.mean(numpy.array(LogDetEstimatesList))

    return LogDet

# =======================
# Compute Log Determinant
# =======================

def ComputeLogDeterminant(A,ComputeMethod='cholesky',NumIterations=20,LanczosDegree=20,UseLanczosTridiagonalization=False):
    """
    Computes the log-determinan of full-rank matrix ``A``.

    :param A: A full-rank matrix.
    :type A: numpy.ndarray or scipy.sparse.csc_matrix

    :param NumIterations: Number of Monte-Carlo trials
    :type NumIterations: int

    :param LanczosDegree: Lanczos degree of the tri-diagonalization (or bi-diaqgonalization) process
    :type LanczosDegree: int

    :param UseLanczosTridiagonalization: Flag, if ``True``, it uses the Lanczos tridiagonalization. 
        If ``False``, it uses the Golub-Kahn bi-diagonalization.
    :type UseLanczosTridiagonalization: bool

    :return: Log-determinant of ``A``
    :rtype: float

    .. note::

        For computing the *trace of inverse* with the sothcastic Lanczos quadrature method 
        (see :mod:`TraceInv.ComputeTraceOfInverse.StochasticLanczosQuadrature`), the preferred algorithm is
        the Lanczos tri-diagonalization, as opposed to Golub-Kahn bi-diagonalization.

        In contrast to the above, the preferred SLQ method for computing *log-determinant* is
        the Golu-Kahn bi-diagonalization. The reason is that if the matrix :math:`\mathbf{A}` 
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

        >>> # Compute log-determinant with stochastic Lanczos quadrature method with Golub-Khan bi-diagonaliation
        >>> LogDet_1 = ComputeLogDeterminant(A,ComputeMethod='SLQ',NumIterations=20, 
        ...                 LanczosDegree=20,Tridiagonalization=False)
    """

    if ComputeMethod == 'cholesky':

        # Cholesky method
        LogDet = CholeskyMethod(A)

    elif ComputeMethod == 'SLQ':

        # Stochastic Lanczos Quadrature method (by Monte-Carlo sampling)
        LogDet = MonteCarloSampling(A,NumIterations,LanczosDegree,UseLanczosTridiagonalization)

    else:
        raise ValueError('ComputeMethod is invalid.')

    return LogDet
