# SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the license found in the LICENSE.txt file in the root
# directory of this source tree.


# =======
# Imports
# =======

import time
import numpy
import scipy
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
from scipy.sparse import isspmatrix
import multiprocessing
from ..__version__ import __version__
from .._linear_algebra.matrix_utilities import get_data_type_name, get_nnz, \
        get_density

try:
    import sksparse
    from sksparse.cholmod import cholesky as sk_cholesky
    suitesparse_installed = True
except ImportError:
    suitesparse_installed = False

# Package
from .._linear_algebra import sparse_cholesky


# ===============
# cholesky method
# ===============

def cholesky_method(
        A,
        gram=False,
        exponent=1,
        invert_cholesky=True,
        cholmod=None):
    """
    Computes trace of inverse of matrix using Cholesky factorization by

    .. math::

        \\mathrm{trace}(\\mathbf{A}^{-1}) = \\| \\mathbf{L}^{-1} \\|_F^2

    where :math:`\\mathbf{L}` is the Cholesky factorization of
    :math:`\\mathbf{A}` and :math:`\\| \\cdot \\|_F` is the Frobenius norm.

    .. note::

        This function does not produce correct results when ``'A'`` is sparse.
        It seems ``sksparse.cholmod`` has a problem.

        When :math:`\\mathbf{A} = \\mathbf{K}` for some positive-definite
        matrix :math:`\\mathbf{K}`, it produces correct result. However, when
        :math:`\\mathbf{A} = \\mathbf{K} + \\eta \\mathbf{I}``, its result is
        different than Hurtchinson and Lanczos stochastic quadrature methods.
        Also its result becomes correct when :math:`\\mathbf{A}` is converted
        to dense matrix, and if we do not use ``skspase.cholmod``.

    :param A: Invertible matrix
    :type A: numpy.ndarray

    :param exponent: Exponent :math:`p` in :math:`\\mathbf{A}^{p}`.
    :param exponent: int

    :param invert_cholesky: Flag to invert Cholesky matrix.
        If ``false``, the inverse of Cholesky is not directly computed, but a
        linear system is solved for each column of the inverse of the Cholesky.
    :type invert_cholesky: bool

    :param cholmod: If set to ``True``, it uses cholmod library from
        scikit-sparse package to compute the Chlesky decomposition. If set to
        ``False``, it uses `scipy.sparse.cholesky`` method. If set to ``None``,
        first, it tries to use cholmod library,  but if cholmod is not
        available, it uses ``scipy.sparse.cholesky`` method without raising any
        warning.

    :return: Trace of matrix ``A``.
    :rtype: float
    """

    # Check input arguments
    check_arguments(A, gram, exponent, invert_cholesky, cholmod)

    # Determine to use Sparse
    sparse = False
    if scipy.sparse.isspmatrix(A):
        sparse = True

    # Determine to use suitesparse or scipy.sparse to compute cholesky
    if suitesparse_installed and cholmod is not False and sparse:
        use_cholmod = True
    else:
        use_cholmod = False

    init_tot_wall_time = time.perf_counter()
    init_cpu_proc_time = time.process_time()

    # Ap is the power of A to the exponent p
    if (exponent == 1) or (exponent == -1):
        if gram:
            Ap = A.T @ A
        else:
            Ap = A

    elif exponent != 0:

        # Initialize Ap
        if gram:
            Ap = A.T @ A
            A1 = Ap.copy()
        else:
            Ap = A.copy()
            A1 = A

        # Directly compute power of A by successive matrix multiplication
        for i in range(1, numpy.abs(exponent)):
            Ap = Ap @ A1

    if exponent == 0:
        trace = A.shape[0]
    elif exponent < 0:

        # Trace of the inverse of a matrix to the power of a negative exponent
        if sparse:
            trace = 0.0
            for i in range(A.shape[0]):
                trace += Ap[i, i]
        else:
            trace = numpy.trace(Ap)

    else:

        # Trace of inverse of matrix to the power of a positive exponent
        # Cholesky factorization
        if sparse:
            if use_cholmod:
                # Using Sparse Suite package
                L = sk_cholesky(Ap)
            else:
                # Using scipy, but with LU instead of Cholesky directly.
                L = sparse_cholesky(Ap)

        else:
            L = scipy.linalg.cholesky(Ap, lower=True)

        # Find Frobenius norm of L inverse
        if invert_cholesky:

            # Invert L directly (better for small matrices)
            trace = compute_traceinv_invert_cholesky_directly(
                    L, sparse, use_cholmod)

        else:
            # Instead of inverting L directly, solve linear system for each
            # column of identity matrix to find columns of the inverse of L
            trace = compute_traceinv_invert_cholesky_indirectly(
                    L, Ap.shape[0], sparse, use_cholmod, A.dtype)

    tot_wall_time = time.perf_counter() - init_tot_wall_time
    cpu_proc_time = time.process_time() - init_cpu_proc_time

    # Dictionary of output info
    info = {
        'matrix':
        {
            'data_type': get_data_type_name(A),
            'gram': gram,
            'exponent': exponent,
            'size': A.shape[0],
            'sparse': isspmatrix(A),
            'nnz': get_nnz(A),
            'density': get_density(A),
            'num_inquiries': 1
        },
        'device':
        {
            'num_cpu_threads': multiprocessing.cpu_count()
        },
        'time':
        {
            'tot_wall_time': tot_wall_time,
            'alg_wall_time': tot_wall_time,
            'cpu_proc_time': cpu_proc_time,
        },
        'solver':
        {
            'version': __version__,
            'method': 'cholesky',
            'invert_cholesky': invert_cholesky,
            'cholmod_used': use_cholmod
        }
    }

    return trace, info


# ===============
# check arguments
# ===============

def check_arguments(A, gram, exponent, invert_cholesky, cholmod):
    """
    Checks the type and value of the parameters.
    """

    # Check A
    if (not isinstance(A, numpy.ndarray)) and (not isspmatrix(A)):
        raise TypeError('Input matrix should be either a "numpy.ndarray" or ' +
                        'a "scipy.sparse" matrix.')
    elif A.shape[0] != A.shape[1]:
        raise ValueError('Input matrix should be a square matrix.')

    # Check gram
    if gram is None:
        raise TypeError('"gram" cannot be None.')
    elif not numpy.isscalar(gram):
        raise TypeError('"gram" should be a scalar value.')
    elif not isinstance(gram, bool):
        raise TypeError('"gram" should be boolean.')

    # Check exponent
    if exponent is None:
        raise TypeError('"exponent" cannot be None.')
    elif not numpy.isscalar(exponent):
        raise TypeError('"exponent" should be a scalar value.')
    elif not isinstance(exponent, (int, numpy.integer)):
        TypeError('"exponent" cannot be an integer.')

    # Check invert_cholesky
    if not numpy.isscalar(invert_cholesky):
        raise TypeError('"invert_cholesky" should be a scalar value.')
    elif invert_cholesky is None:
        raise ValueError('"invert_cholesky" cannot be None.')
    elif not isinstance(invert_cholesky, bool):
        raise TypeError('"invert_cholesky" should be an integer.')

    # Check cholmod
    if cholmod is not None:
        if not isinstance(cholmod, bool):
            raise TypeError('"cholmod" should be either "None", or boolean.')
        elif cholmod is True and suitesparse_installed is False:
            raise RuntimeError('"cholmod" method is not available. Either ' +
                               'install "scikit-sparse" package, or set ' +
                               '"cholmod" to "False" or "None".')


# =========================================
# compute traceinv invert cholesky directly
# =========================================

def compute_traceinv_invert_cholesky_directly(L, sparse, use_cholmod):
    """
    Compute the trace of inverse by directly inverting the Cholesky matrix
    :math:`\\mathbb{L}`.

    .. note::

        * For small matrices: This method is much faster for small matrices
          than :py:func:`compute_traceinv_invert_cholesky_indirectly`.
        * For large matrices: This method is very slow and results are
          unstable.

    .. warning::

        If scikit-sparse package is used to compute Cholesky decomposition,
        all computations are done using ``float64`` data type. The 32-bit type
        is not available in that package.

    :param L: Cholesky matrix
    :type L: numpy.ndarray

    :param sparse: Flag, if ``true``, the matrix `L`` is considered as sparse.
    :type sparse: bool

    :param use_cholmod: If ``True``, uses ``scikit-sparse`` package to compute
        the Cholesky decomposition. If ``False``, uses ``scipy.sparse``
        package.
    :type use_cholmod: bool

    :return: Trace of matrix ``A``.
    :rtype: float
    """

    # Direct method. Take inverse of L, then compute its Frobenius norm.
    if sparse:

        if use_cholmod:

            # Using cholmod. Note, here L_ is the decomposition L_ L_.T = A,
            # not L_ D L_.T = A.
            L_ = L.L()
            Linv = scipy.sparse.linalg.inv(L_)

        else:
            # Using scipy to compute inv of cholesky
            Linv = scipy.sparse.linalg.inv(L)

        trace = scipy.sparse.linalg.norm(Linv, ord='fro')**2
    else:
        # Dense matrix
        Linv = scipy.linalg.inv(L)
        trace = numpy.linalg.norm(Linv, ord='fro')**2

    return trace


# ===========================================
# compute traceinv invert cholesky indirectly
# ===========================================

def compute_traceinv_invert_cholesky_indirectly(
        L, n, sparse, use_cholmod, dtype):
    """
    Computes the trace of inverse by solving a linear system for Cholesky
    matrix and each column of the identity matrix to obtain the inverse of
    ``L`` sub-sequentially.

    The matrix :math:`\\mathbf{L}` is not inverted directly, rather, the linear
    system

    .. math::

        \\mathbf{L} \\boldsymbol{x}_i =
        \\boldsymbol{e}_i, \\qquad i = 1,\\dots,n

    is solved, where
    :math:`\\boldsymbol{e}_i = (0, \\dots, 0, 1, 0, \\dots, 0)^{\\intercal}` is
    a column vector of zeros except its :math:`i`:superscript:`th` entry is one
    and :math:`n` is the size of the square matrix :math:`\\mathbf{A}`. The
    solution :math:`\\boldsymbol{x}_i` is the :math:`i`:superscript:`th` column
    of :math:`\\mathbf{L}^{-1}`. Then, its Frobenius norm is

    .. math::

        \\| \\mathbf{L} \\|_F^2 = \\sum_{i=1}^n \\| \\boldsymbol{x}_i \\|^2.

    The method is memory efficient as the vectors :math:`\\boldsymbol{x}_i` do
    not need to be stored, rather, their norm can be stored in each iteration.

    .. note::

        This method is slow, and it should be used only if the direct matrix
        inversion can not be computed (such as for large matrices).

    .. warning::

        If scikit-sparse package is used to compute Cholesky decomposition,
        all computations are done using ``float64`` data type. The 32-bit type
        is not available in that package.

    :param L: Cholesky matrix
    :type L: numpy.ndarray

    :param sparse: Flag, if ``true``, the matrix ``L`` is considered as sparse.
    :type sparse: bool

    :param use_cholmod: If ``True``, uses ``scikit-sparse`` package to compute
        the Cholesky decomposition. If ``False``, uses ``scipy.sparse``
        package.
    :type use_cholmod: bool

    :param dtype: The data type of matrix.
    :type dtype: string or numpy.dtype

    :return: Trace of matrix ``A``.
    :rtype: float
    """

    # Instead of finding L inverse, and then its norm, we directly find norm
    norm2 = 0

    # Solve a linear system that finds each of the columns of L inverse
    for i in range(n):

        # Handle sparse matrices
        if sparse:

            # e is a zero vector with its i-th element is one
            e = scipy.sparse.lil_matrix((n, 1), dtype=dtype)
            e[i] = 1.0

            # x solves of L x = e. Thus, x is the i-th column of L inverse.
            if use_cholmod and \
               isinstance(L, sksparse.cholmod.Factor):

                # Using cholmod.Note: LDL SHOULD be disabled.
                x = L.solve_L(
                        e.tocsc(),
                        use_LDLt_decomposition=False).toarray()

            elif scipy.sparse.isspmatrix(L):

                # Using scipy
                x = scipy.sparse.linalg.spsolve_triangular(
                        L.tocsr(),
                        e.toarray(),
                        lower=True)

            else:
                raise RuntimeError('Unknown sparse matrix type.')

            # Append to the Frobenius norm of L inverse
            norm2 += numpy.sum(x**2)

        else:

            # e is a zero vector with its i-th element is one
            e = numpy.zeros(n, dtype=dtype)
            e[i] = 1.0

            # x solves L * x = e. Thus, x is the i-th column of L inverse
            x = scipy.linalg.solve_triangular(L, e, lower=True)

            # Append to the Frobenius norm of L inverse
            norm2 += numpy.sum(x**2)

    trace = norm2

    return trace
