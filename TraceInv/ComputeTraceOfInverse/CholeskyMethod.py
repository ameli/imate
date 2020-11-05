# =======
# Imports
# =======

import numpy
import scipy
from scipy import sparse
from scipy import linalg
from scipy.sparse import linalg

try:
    import sksparse
    from sksparse.cholmod import cholesky
    SparseSuiteInstalled = True
except:
    SparseSuiteInstalled = False


# Package
from .LinearSolver import LinearSolver

# ========================================
# Compute Trace Inv By Inverse of Cholesky
# ========================================

def ComputeTraceInvByInverseOfCholesky(L,UseSparse):
    """
    Compute the trace of inverse by inverting the Cholesky matrix ``L`` directly.

    .. note::

        * For small matrices: This method is much faster for small matrices than :py:func:`ComputeTraceInvBySolvingLinearSystem`.
        * For large matrices: This method is very slow and results are unstable.

    :param L: Cholesky matrix
    :type L: ndarray

    :param UseSparse: Flag, if ``true``, the matrix ``L`` is considered as sparse.
    :type UseSparse: bool

    :return: Trace of matrix ``A``.
    :rtype: float
    """

    # Direct method. Take inverse of L, then compute its Frobenius norm.
    if UseSparse:

        raise ValueError('Do not use sksparse.cholmod.inv, as it computes LDLt decomposition and the computed trace becomes incorrect. Either set UseInverseMatrix to False when using sparse matrices, or use Hutchinson or Lanczos method.')

        # Note: the L.inv() uses LDLt decomposition, not LLt, which then the compueted Trace becomes incorrect.
        Linv = L.inv()
        Trace = scipy.sparse.linalg.norm(Linv,ord='fro')**2
    else:
        Linv = scipy.linalg.inv(L)
        Trace = numpy.linalg.norm(Linv,ord='fro')**2

    return Trace

# ==========================================
# Compute Trace Inv By Solving Linear System
# ==========================================

def ComputeTraceInvBySolvingLinearSystem(L,n,UseSparse):
    """
    Computes the trace of inverse by solving a linear system for Cholesky matrix and each column of the identity matrix
    to obtain the inverse of ``L`` subsequentially.

    .. note::

        * For small matrices: This method is slow.
        * For large matrices: This method should be used for large matrices.

    :param L: Cholesky matrix
    :type L: ndarray

    :param UseSparse: Flag, if ``true``, the matrix ``L`` is considered as sparse.
    :type UseSparse: bool

    :return: Trace of matrix ``A``.
    :rtype: float
    """

    # Instead of finding L inverse, and then its norm, we directly find norm
    Norm2 = 0

    # Solve a linear system that finds each of the columns of L inverse
    for i in range(n):

        # Handle sparse matrices
        if UseSparse:

            # e is a zero vector with its i-th element is one
            e = scipy.sparse.lil_matrix((n,1),dtype=float)
            e[i] = 1.0

            # x is the solution of L x = e. Thus, x is the i-th column of L inverse.
            if SparseSuiteInstalled and isinstance(L,sksparse.cholmod.Factor):

                # Using cholmod.Note: LDL SHOULD be disabled.
                x = L.solve_L(e.tocsc(),use_LDLt_decomposition=False).toarray()

            elif isinstance(L,scipy.sparse.csc.csc_matrix):

                # Using scipy
                x = scipy.sparse.linalg.spsolve_triangular(L.tocsr(),e.toarray(),lower=True)

            else:
                raise RuntimeError('Unknown sparse matrix type.')

            # Append to the Frobenius norm of L inverse
            Norm2 += numpy.sum(x**2)

        else:

            # e is a zero vector with its i-th element is one
            e = numpy.zeros(n)
            e[i] = 1.0

            # x is the solution of L x = e. Thus, x is the i-th column of L inverse
            x = scipy.linalg.solve_triangular(L,e,lower=True)

            # Append to the Frobenius norm of L inverse
            Norm2 += numpy.sum(x**2)

    Trace = Norm2

    return Trace

# ===============
# Sparse Cholesky
# ===============

def SparseCholesky(A):
    """
    This function uses LU decomposition assuming that A is symmetric and positive-definite.

    .. note::

        This function does not check if ``A`` is positive-definite. If the input matrix is 
        not positive-definite, the Cholesky decomposition does not exist and the return value
        is misleadingly wrong.

    :param A: Symmetric and positive-definite matrix.
    :type A: ndarray

    :return: Chlesky decomposition of ``A``.
    :rtype: Super.LU
    """

    n = A.shape[0]

    # sparse LU decomposition
    LU = scipy.sparse.linalg.splu(A,diag_pivot_thresh=0,permc_spec='NATURAL')

    return LU.L.dot(sparse.diags(LU.U.diagonal()**0.5))

    # check the matrix A is positive definite.
    if (LU.perm_r == numpy.arange(n)).all() and (LU.U.diagonal() > 0).all():
        return LU.L.dot(sparse.diags(LU.U.diagonal()**0.5))
    else:
        raise RuntimeError('The matrix is not positive definite.')

# ===============
# Cholesky Method
# ===============

def CholeskyMethod(A,UseInverseMatrix=True):
    """
    Computes trace of inverse of matrix using Cholesky factorization.

    .. math::

        \\mathrm{trace}(\\mathbf{A}^{-1}) = \\| \\mathbf{L}^{-1} \\|_F^2

    where :math:`\\mathbf{L}` is the Cholesky factorization of :math:`\\mathbf{A}`, 
    and :math:`\\| \\cdot \\|_F` is the Frobenius norm.

    .. note::

        This function does not produce correct results when ``'A'`` is sparse.
        It seems ``sksparse.cholmod`` has a problem.

        When ``A = K``, it produces correct result, But when ``A = Kn = K + eta I``, its result is
        differen than Hurchinson, Lanczos method. Also its result becomes correct when ``A`` is converted
        to dense matrix, and when we do not use ``skspase.cholmod`` anymore.

    :param A: Invertible matrix
    :type A: ndarray

    :param UseInverseMatrix: Flag to invert Cholesky matrix.
        If ``false``, the inverse of Cholesky is not computed, but a linear system is solved.
    :type UseInverseMatrix: bool

    :return: Trace of matrix ``A``.
    :rtype: float
    """

    # Determine to use Sparse
    UseSparse = False
    if scipy.sparse.isspmatrix(A):
        UseSparse = True

    # Cholesky factorization
    if UseSparse:
        try:
            # Using Sparse Suite package
            L = sksparse.cholmod.cholesky(A)
        except:
            # Using scipy, but with LU instad of Cholesky directoy
            L = SparseCholesky(A)
            # raise RuntimeError('The package "sksparse" is not installed. '
            #     'Either install "sksparse", or do not use Cholesky method for sparse matrices. '
            #     'Alternative methods are Hutchinson method and stochastic Lanczos quadrature methods.')
            
    else:
        L = scipy.linalg.cholesky(A,lower=True)

    # Find Frobenius norm of the inverse of L. If matrix size is small, compute inverse directly
    if UseInverseMatrix == True:

        # Invert L directly
        Trace = ComputeTraceInvByInverseOfCholesky(L,UseSparse)

    else:
        # Instead of inverting L, solve linear system for each column of identity matrix
        Trace = ComputeTraceInvBySolvingLinearSystem(L,A.shape[0],UseSparse)

    return Trace
