# =======
# Imports
# =======

import numpy
cimport numpy
from numpy import linalg
import scipy.sparse
import multiprocessing
cimport cython
from cython import boundscheck,wraparound
from cython.parallel import parallel, prange
from libc.stdio cimport fflush,stdout,printf

try:
    cimport openmp
    OpenMPInstalled = True
except ImportError:
    OpenMPInstalled = False

from .._LinearAlgebra cimport LanczosTridiagonalization
from .._LinearAlgebra cimport GolubKahnBidiagonalization
from .._LinearAlgebra cimport CreateBandMatrix

# ====================================
# Stochastic Lanczos Quadrature Method
# ====================================

def StochasticLanczosQuadratureMethod(
        A,
        NumIterations=20,
        LanczosDegree=20,
        UseLanczosTridiagonalization=False):
    """
    Computes the trace of inverse of matrix based on stochastic Lanczos quadrature method.

    :param A: invertible matrix
    :type A: ndarray

    :param NumIterations: Number of Monte-Carlo trials
    :type NumIterations: int

    :param LanczosDegree: Lanczos degree
    :type LanczosDegree: int

    :param UseLanczosTridiagonalization: Flag, if ``True``, it uses the Lanczos tri-diagonalization. 
        If ``False``, it uses the Golub-Kahn bi-diagonalization.
    :type UseLanczosTridiagonalization: bool

    :return: Trace of ``A``
    :rtype: float

    This function is a wrapper to a cython function. In cython we cannot have function arguments 
    with default values (neither cdef, cpdef). As a work around, this function (defined with ``def``)
    accepts default values for arguments, and then calls the cython function (defined with ``cdef``).

    .. note::

        In Lanczos tri-diagonalization method, :math:`\\theta`` is the eigenvalue of ``T``. 
        However, in Golub-Kahn bi-diagonalization method, :math:`\\theta` is the singular values of ``B``.
        The relation between these two methods are are follows: ``B.T*B`` is the ``T`` for ``A.T*A``.
        That is, if we have the input matrix ``A.T*T``, its Lanczos tri-diagonalization ``T`` is the same matrix
        as if we bi-diagonalize ``A`` (not ``A.T*A``) with Golub-Kahn to get ``B``, then ``T = B.T*B``.
        This has not been highlighted paper in the above paper.

        To correctly implement Golub-Kahn, here :math:`\\theta` should be the singular values of ``B``, **NOT**
        the square of the singular values of ``B`` (as described in the above paper incorrectly!).

    Reference:
        * `Ubaru, S., Chen, J., and Saad, Y. (2017) <https://www-users.cs.umn.edu/~saad/PDF/ys-2016-04.pdf>`_, 
          Fast Estimation of :math:`\\mathrm{tr}(F(A))` Via Stochastic Lanczos Quadrature, SIAM J. Matrix Anal. Appl., 38(4), 1075-1099.
    """

    # Check if A is sparse or dense
    if scipy.sparse.issparse(A):

        # Check sorted indices
        if not A.has_sorted_indices:
            raise RuntimeError('Sparse matrix A should have sorted indices.')

        # Check CSR format
        if not isinstance(A,scipy.sparse.csr.csr_matrix):

            print('To increase performance, it is recommended to provide the input matrix as "CSR" format.')
            A = A.tocsr()

        # Calling a cython function without default arguments
        return _StochasticLanczosQuadratureMethod(
                None,
                A.data,
                A.indices,
                A.indptr,
                NumIterations,
                LanczosDegree,
                int(UseLanczosTridiagonalization))

    else:

        # Dense matrix. Calling a cython function without default arguments
        return _StochasticLanczosQuadratureMethod(
                A,
                None,
                None,
                None,
                NumIterations,
                LanczosDegree,
                int(UseLanczosTridiagonalization))

# =============================================
# Cython's Stochastic Lanczos Quadrature Method
# =============================================

cpdef double _StochasticLanczosQuadratureMethod(
        const double[:,::1] A,
        const double[:] A_Data,
        const int[:] A_ColumnIndices,
        const int[:] A_IndexPointer,
        const int NumIterations,
        const int LanczosDegree,
        const int UseLanczosTridiagonalization) except *:
    """
    This is the actual implementation of the SLQ method, but without the
    default values for the function argument, since it is not possible in cython.

    :param A: invertible matrix
    :type A: cython memortview

    :param NumIterations: Number of Monte-Carlo trials
    :type NumIterations: int

    :param LanczosDegree: Lanczos degree
    :type LanczosDegree: int

    :param UseLanczosTridiagonalization: Flag, if ``True``, it uses the Lanczos tri-diagonalization. 
        If ``False``, it uses the Golub-Kahn bi-diagonalization.
    :type UseLanczosTridiagonalization: int

    :return: Trace of ``A``
    :rtype: float
    """

    # Typed variables
    cdef int UseSparse
    cdef int n
    cdef int i
    cdef double Trace
    cdef double[:] Theta
    cdef double[:] Tau2
    cdef double Tolerance = 1e-8

    # Check if the input matrix is sparse
    if A is None:
        if (A_Data is None) or (A_ColumnIndices is None) or (A_IndexPointer is None):
            raise ValueError('All components of sparse matrix are not provided.')
        else:
            UseSparse = 1
    else:
        UseSparse = 0

    # Get the size of matrix
    if UseSparse:
        n = A_IndexPointer.size - 1
    else:
        n = A.shape[0]

    # Allocate counter during Lanczos iterations, which determines the size of matrix tri-diagonal (or bi-diagonal) matrix
    cdef int[:] LanczosCounters = numpy.zeros((NumIterations,),dtype=numpy.int32)

    # Allocate the array to hold the results of Monte-Carlo trials
    cdef double[:] TraceEstimates = numpy.zeros((NumIterations,),dtype=float)

    # Radamacher random vector, consists of 1 and -1 (this is 2D array to hold all trials)
    cdef double[:,::1] w = numpy.sign(numpy.random.randn(NumIterations,n))

    # diagonals and off-diagonals of tri-diagonal (or bi-diagonal) matrix T
    cdef double[:,::1] alphas = numpy.zeros((NumIterations,LanczosDegree),dtype=float)    # diagonals
    cdef double[:,::1] betas = numpy.zeros((NumIterations,LanczosDegree+1),dtype=float)   # off-diagonals

    # Set the number of threads
    cdef int NumberOfParallelThreads = multiprocessing.cpu_count()
    # if OpenMPInstalled:
    #     openmp.omp_set_num_threads(NumberOfParallelThreads)

    # Chunk size for parallel schedule, using square root of max possible chunk size
    cdef int ChunkSize = int(NumIterations / NumberOfParallelThreads)
    if ChunkSize < 1:
        ChunkSize = 1
    
    # Shared-memory parallelism over Monte-Carlo iterations. This fills diagonals (alphas) and off-diagonals (betas)
    with nogil, parallel():
        for i in prange(NumIterations,schedule='dynamic',chunksize=ChunkSize):

            if UseLanczosTridiagonalization:

                # Use Lanczos Tri-diagonalization
                LanczosCounters[i] = LanczosTridiagonalization(
                        A,
                        A_Data,
                        A_ColumnIndices,
                        A_IndexPointer,
                        w[i,:],n,LanczosDegree,Tolerance,alphas[i,:],betas[i,:])

            else:
                # Use Golub-Kahn-Lanczos Bi-diagonalization
                LanczosCounters[i] = GolubKahnBidiagonalization(
                        A,
                        A_Data,
                        A_ColumnIndices,
                        A_IndexPointer,
                        w[i,:],n,LanczosDegree,Tolerance,alphas[i,:],betas[i,:])

    # Allocate tri-diagonal (or bi-diagonal) matrix
    cdef double[:,::1] T = numpy.zeros((LanczosDegree,LanczosDegree),dtype=float)

    # Back to using python gil. Create tri-diagonal (or bi-diagonal) matrix for each trial
    for i in range(NumIterations):

        # Choose between Lanczos tri-diagonalization or Golub-Kahn bi-diagonalization
        if UseLanczosTridiagonalization:

            # Create a tri-diagonal matrix from the bands alpha and beta
            CreateBandMatrix(alphas[i,:],betas[i,:],LanczosCounters[i],UseLanczosTridiagonalization,T)

            # Spectral decomposition of T
            Eigenvalues,Eigenvectors = numpy.linalg.eigh(T[:LanczosCounters[i],:LanczosCounters[i]])

            # Theta and Tau
            Theta = numpy.abs(Eigenvalues)
            Tau2 = Eigenvectors[0,:]**2

        else:

            # Create a bi-diagonal matrix from the bands alpha and beta
            CreateBandMatrix(alphas[i,:],betas[i,:],LanczosCounters[i],UseLanczosTridiagonalization,T)

            # Singular value decomposition of the bi-diagonal matrix T
            LeftEigenvectors,SingularValues,RightEigenvectorsTransposed = numpy.linalg.svd(T[:LanczosCounters[i],:LanczosCounters[i]])

            # Theta and Tau
            Theta = SingularValues    # Note: Theta here is just singular values, not singular values squared
            Tau2 = RightEigenvectorsTransposed[:,0]**2

        # Quadrature summation
        TraceEstimates[i] = 0.0
        for j in range(LanczosDegree):

            # Here, f(theta) = 1/theta, since we compute trace of matrix inverse
            TraceEstimates[i] = TraceEstimates[i] + n*Tau2[j]/Theta[j]

    Trace = numpy.mean(TraceEstimates)

    return Trace
