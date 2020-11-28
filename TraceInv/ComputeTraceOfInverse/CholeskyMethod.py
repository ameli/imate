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
    SuiteSparseInstalled = True
except:
    SuiteSparseInstalled = False

# Package
from ..LinearAlgebra import LinearSolver
from ..LinearAlgebra import SparseCholesky

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

        # Note: the L.inv() uses LDLt decomposition, not LLt, which then the computed Trace becomes incorrect.
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
    to obtain the inverse of ``L`` sub-sequentially.

    The matrix :math:`\mathbf{L}` is not inverted directly, rather, the linear system

    .. math::

        \mathbf{L} \\boldsymbol{x}_i = \\boldsymbol{e}_i, \qquad i = 1,\dots,n

    is solved, where :math:`\\boldsymbol{e}_i = (0,\dots,0,1,0,\dots,0)^{\intercal}` is a column vector of zeros except its
    :math:`i` th entry is one, and :math:`n` is the size of the square matrix :math:`\mathbf{A}`. 
    The solution :math:`\\boldsymbol{x}_i` is the :math:`i` th column of :math:`\mathbf{L}^{-1}`. Then, its Frobenius norm is

    .. math::

        \| \mathbf{L} \|_F^2 = \sum_{i=1}^n \| \\boldsymbol{x}_i \|^2.

    The method is memory efficient as the vectors :math:`\\boldsymbol{x}_i` do not need to be stored, 
    rather, their norm can be stored in each iteration.

    .. note::

        This method is slow, and it should be used only if the direct matrix inversion can not
        be computed (such as for large matrices).

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
            if SuiteSparseInstalled and isinstance(L,sksparse.cholmod.Factor):

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
# Cholesky Method
# ===============

def CholeskyMethod(A,UseInverseMatrix=True):
    """
    Computes trace of inverse of matrix using Cholesky factorization by

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
            # Using scipy, but with LU instead of Cholesky directly.
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
