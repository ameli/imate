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
        p=1,
        return_info=False,
        B=None,
        invert_cholesky=True,
        cholmod=None):
    """
    Log-determinant of non-singular matrix using Cholesky method.

    Given the matrix :math:`\\mathbf{A}` and the real exponent :math:`p`, the
    following is computed:

    .. math::

        \\mathrm{logdet} \\left(\\mathbf{A}^p \\right) = p \\log_e \\vert
        \\det (\\mathbf{A}) \\vert.

    If ``gram`` is `True`, then :math:`\\mathbf{A}` in the above is replaced by
    the Gramian matrix :math:`\\mathbf{A}^{\\intercal} \\mathbf{A}`, and the
    following is instead computed:

    .. math::

        \\mathrm{logdet} \\left((\\mathbf{A}^{\\intercal}\\mathbf{A})^p
        \\right) = 2p \\log_e \\vert \\det (\\mathbf{A}) \\vert.

    Parameters
    ----------

    A : numpy.ndarray, scipy.sparse
        A non-singular sparse or dense matrix. For ``method=cholesky``, the
        matrix `A` should be positive-definite.

        .. note::

            In the Cholesky method, the matrix cannot be a type of
            :class:`Matrix` or :class:`imate.AffineMatrixFunction` classes.

    gram : bool, default=False
        If `True`, the log-determinant of the Gramian matrix,
        :math:`(\\mathbf{A}^{\\intercal}\\mathbf{A})^p`, is computed. The
        Gramian matrix itself is not directly computed. If `False`, the
        log-determinant of :math:`\\mathbf{A}^p` is computed.

    p : float, default=1.0
        The exponent :math:`p` in :math:`\\mathbf{A}^p`.

    return_info : bool, default=False
        If `True`, this function also returns a dictionary containing
        information about the inner computation, such as process time,
        algorithm settings, etc.

    :param invert_cholesky: Flag to invert Cholesky matrix.
        If ``false``, the inverse of Cholesky is not directly computed, but a
        linear system is solved for each column of the inverse of the Cholesky.
    :type invert_cholesky: bool

    cholmod : bool, default=None
        If set to `True`, it uses the `Cholmod` library from `scikit-sparse`
        package to compute the Cholesky decomposition. If set to `False`, it
        uses `scipy.sparse.cholesky` method. If set to `None`, first, it tries
        to use Cholmod library,  but if Cholmod is not available, it uses
        `scipy.sparse.cholesky` method.

    Returns
    -------

    logdet : float or numpy.array
        Log-determinant of `A`.

    info : dict
        (Only if ``return_info`` is `True`) A dictionary of information with
        the following keys.

        * ``matrix``:
            * ``data_type``: `str`, {`float32`, `float64`, `float128`}. Type of
              the matrix data.
            * ``gram``: `bool`, whether the matrix `A` or its Gramian is
              considered.
            * ``exponent``: `float`, the exponent `p` in :math:`\\mathbf{A}^p`.
            * ``size``: `int`, the size of matrix `A`.
            * ``sparse``: `bool`, whether the matrix `A` is sparse or dense.
            * ``nnz``: `int`, if `A` is sparse, the number of non-zero elements
              of `A`.
            * ``density``: `float`, if `A` is sparse, the density of `A`, which
              is the `nnz` divided by size squared.
            * ``num_inquiries``: `int`, for the `cholesky` method, this is
              always `1`.

        * ``device``:
            * ``num_cpu_threads``: `int`, number of CPU threads used in shared
              memory parallel processing.
            * ``num_gpu_devices``: `int`, for the `cholesky` method, this is
              always `0`.
            * ``num_gpu_multiprocessors``: `int`, for the `cholesky` method,
              this is always `0`.
            * ``num_gpu_threads_per_multiprocessor``: `int`, for the `cholesky`
              method, this is always `0`.

        * ``time``:
            * ``tot_wall_time``: `float`, total elapsed time of computation.
            * ``alg_wall_time``: `float`, elapsed time of computation during
              only the algorithm execution.
            * ``cpu_proc_time``: `float`, CPU processing time of computation.

        * ``solver``:
            * `cholmod_used`: `bool`, whether the Cholmod from SparseSuite
              library was used.
            * ``version``: `str`, version of imate.
            * ``method``: 'cholesky'

    See Also
    --------

    imate.trace
    imate.traceinv
    imate.schatten

    Notes
    -----

    The log-determinant is computed from the Cholesky decomposition
    :math:`\\mathbf{A} = \\mathbf{L} \\mathbf{L}^{\\intercal}` as

    .. math::

        \\log | \\mathbf{A} | =
        2 \\mathrm{trace}( \\log \\mathrm{diag}(\\mathbf{L})).


    This function is essentially a wrapper for the Cholesky function of the
    `scipy` and `scikit-sparse` packages and is primarily used for testing and
    comparison (benchmarking) against the randomized methods that are
    implemented in this package. If ``cholmod`` is set to `True`, this function
    uses the `Suite Sparse
    <https://people.engr.tamu.edu/davis/suitesparse.html>`_ package to compute
    the Cholesky decomposition.

    The result is exact (no approximation) and could be used as a benchmark to
    test other methods.

    .. note::

        This function does not produce correct results when ``'A'`` is sparse.
        It seems ``sksparse.cholmod`` has a problem.

        When :math:`\\mathbf{A} = \\mathbf{K}` for some positive-definite
        matrix :math:`\\mathbf{K}`, it produces correct result. However, when
        :math:`\\mathbf{A} = \\mathbf{K} + \\eta \\mathbf{I}``, its result is
        different than Hurtchinson and Lanczos stochastic quadrature methods.
        Also its result becomes correct when :math:`\\mathbf{A}` is converted
        to dense matrix, and if we do not use ``skspase.cholmod``.


    Examples
    --------

    Compute the log-determinant of a sparse positive-definite Toeplitz matrix:

    .. code-block:: python

        >>> # Import packages
        >>> from imate import toeplitz, logdet

        >>> # Generate a sample symmetric and positive-definite matrix
        >>> A = toeplitz(2, 1, size=100, gram=True)

        >>> # Compute log-determinant with Cholesky method (default method)
        >>> logdet(A, method='cholesky')
        138.62943611198907

    Print information about the inner computation:

    .. code-block:: python

        >>> ld, info = logdet(A, method='cholesky', return_info=True)
        >>> print(ld)
        138.6294361119891

        >>> # Print dictionary neatly using pprint
        >>> from pprint import pprint
        >>> pprint(info)
        {
            'matrix': {
                'data_type': b'float64',
                'density': 0.0298,
                'exponent': 1.0,
                'gram': False,
                'nnz': 298,
                'num_inquiries': 1,
                'size': 100,
                'sparse': True
            },
            'device': {
                'num_cpu_threads': 8,
                'num_gpu_devices': 0,
                'num_gpu_multiprocessors': 0,
                'num_gpu_threads_per_multiprocessor': 0
            },
            'solver': {
                'cholmod_used': True,
                'method': 'cholesky',
                'version': '0.13.0'
            },
            'time': {
                'alg_wall_time': 0.0007234140066429973,
                'cpu_proc_time': 0.0009358710000000325,
                'tot_wall_time': 0.0007234140066429973
            }
        }
    """

    # Check input arguments
    check_arguments(A, B, gram, p, invert_cholesky, cholmod)

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

    # Form A**p (or (AtA)**p), that is the p-th power of A or (A.T * A)
    if (p == 1) or (p == -1):
        if gram:
            Ap = A.T @ A
        else:
            Ap = A

    elif p != 0:

        # Initialize Ap
        if gram:
            Ap = A.T @ A
            A1 = Ap.copy()
        else:
            Ap = A.copy()
            A1 = A

        # Directly compute power of A by successive matrix multiplication
        for i in range(1, numpy.abs(p)):
            Ap = Ap @ A1

    # Compute traceinv
    if p == 0:
        if B is None:
            trace = A.shape[0]
        else:
            if isspmatrix(B):
                trace = 0
                for i in range(B.shape[0]):
                    trace += B[i, i]
            else:
                trace = numpy.trace(B)
    elif p < 0:

        if B is None:
            C = Ap
        else:
            C = Ap @ B

        # Trace of the inverse of a matrix to the power of a negative p
        if sparse:
            trace = 0.0
            for i in range(C.shape[0]):
                trace += C[i, i]
        else:
            trace = numpy.trace(C)

    else:

        # Trace of inverse of matrix to the power of a positive p
        # Cholesky factorization
        if sparse:

            if use_cholmod:
                # Using Sparse Suite package. Using default ordering mode.
                # There is a non-trivial permutation matrix P associated with
                # the Cholesky decomposition L_A.
                L_A = sk_cholesky(Ap, ordering_method='default')

                # L_B is the Cholesky decomposition of B
                if B is None:
                    L_B = None
                else:
                    # Using natural ordering mode, hence there is no
                    # permutation matrix P associated with the Cholesky
                    # decomposition L_B.
                    L_B = sk_cholesky(B, ordering_method='natural').L()
            else:
                # Using scipy, but with LU instead of Cholesky directly.
                L_A = sparse_cholesky(Ap)

                # Cholesky of B
                if B is None:
                    L_B = None
                else:
                    L_B = sparse_cholesky(B)

        else:
            L_A = scipy.linalg.cholesky(Ap, lower=True)

            # Cholesky of B
            if B is None:
                L_B = None
            else:
                L_B = scipy.linalg.cholesky(B, lower=True)

        # Find Frobenius norm of L_A inverse
        if invert_cholesky:

            # Invert L_A directly (better for small matrices)
            trace = compute_traceinv_invert_cholesky_directly(
                    L_A, L_B, sparse, use_cholmod)

        else:
            # Instead of inverting L_A directly, solve linear system for each
            # column of identity matrix to find columns of the inverse of L_A
            trace = compute_traceinv_invert_cholesky_indirectly(
                    L_A, L_B, Ap.shape[0], sparse, use_cholmod, A.dtype)

    tot_wall_time = time.perf_counter() - init_tot_wall_time
    cpu_proc_time = time.process_time() - init_cpu_proc_time

    # Dictionary of output info
    info = {
        'matrix':
        {
            'data_type': get_data_type_name(A),
            'gram': gram,
            'exponent': p,
            'size': A.shape[0],
            'sparse': isspmatrix(A),
            'nnz': get_nnz(A),
            'density': get_density(A),
            'num_inquiries': 1
        },
        'device':
        {
            'num_cpu_threads': multiprocessing.cpu_count(),
            'num_gpu_devices': 0,
            'num_gpu_multiprocessors': 0,
            'num_gpu_threads_per_multiprocessor': 0
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

    if return_info:
        return trace, info
    else:
        return trace


# ===============
# check arguments
# ===============

def check_arguments(A, B, gram, p, invert_cholesky, cholmod):
    """
    Checks the type and value of the parameters.
    """

    # Check A
    if (not isinstance(A, numpy.ndarray)) and (not isspmatrix(A)):
        raise TypeError('Input matrix should be either a "numpy.ndarray" or ' +
                        'a "scipy.sparse" matrix.')
    elif A.shape[0] != A.shape[1]:
        raise ValueError('Input matrix should be a square matrix.')

    # Check B
    if B is not None:
        if (isinstance(A, numpy.ndarray)) and \
                (not isinstance(B, numpy.ndarray)):
            raise TypeError('When the input matrix "A" is of type ' +
                            '"numpy.ndarray", matrix "B" should also be of ' +
                            'the same type.')
        if isspmatrix(A) and not isspmatrix(B):
            raise TypeError('When the input matrix "A" is of type ' +
                            '"scipy.sparse", matrix "B" should also be of ' +
                            'the same type.')
        elif A.shape != B.shape:
            raise ValueError('Matrix "B" should have the same size as ' +
                             'matrix "A".')

    # Check gram
    if gram is None:
        raise TypeError('"gram" cannot be None.')
    elif not numpy.isscalar(gram):
        raise TypeError('"gram" should be a scalar value.')
    elif not isinstance(gram, bool):
        raise TypeError('"gram" should be boolean.')

    # Check p
    if p is None:
        raise TypeError('"p" cannot be None.')
    elif not numpy.isscalar(p):
        raise TypeError('"p" should be a scalar value.')
    elif not isinstance(p, (int, numpy.integer)):
        raise TypeError('"p" should be an integer.')

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

def compute_traceinv_invert_cholesky_directly(L_A, L_B, sparse, use_cholmod):
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

    :param L_A: Cholesky factorization of matrix A
    :type L_A: numpy.ndarray, scipy.sprase matrix, or sksparse.cholmod.Factor

    :param L_B: Cholesky factorization of matrix B. If set to None, it is
        assumed that matrix B, and hence L_B, is identify.
    :type L_B: numpy.ndarray, scipy.sprase matrix, or sksparse.cholmod.Factor

    :param sparse: Flag, if ``true``, the matrix `L`` is considered as sparse.
    :type sparse: bool

    :param use_cholmod: If ``True``, uses ``scikit-sparse`` package to compute
        the Cholesky decomposition. If ``False``, uses ``scipy.sparse``
        package.
    :type use_cholmod: bool

    :return: Trace of matrix ``A``.
    :rtype: float
    """

    # Direct method. Take inverse of L_A, then compute its Frobenius norm.
    if sparse:

        if use_cholmod:

            # Note: here, L_A_ is the Cholesky decomposition of A in the form
            # of L_A_ * L_A_.T = A, and not L_A_ * D * L_A_.T = A.
            L_A_ = L_A.L()
            L_A_inv = scipy.sparse.linalg.inv(L_A_)

        else:
            # Using scipy to compute inv of cholesky
            L_A_inv = scipy.sparse.linalg.inv(L_A)

        # Multiply by L_B
        if L_B is not None:

            # Cholesky decomposition with scikit-sparse has non-trivial
            # permutation matrix P. Left multiplication by P permutes the rows
            # of matrix. P is a row vector.
            if sparse and use_cholmod and \
                    isinstance(L_A, sksparse.cholmod.Factor):
                P = L_A.P()
                C = L_A_inv @ L_B[P, :]
            else:
                # No cholmod is used. Cholesky decomposition of A does has the
                # natural permutation (no permutation).
                C = L_A_inv @ L_B
        else:
            C = L_A_inv

        trace = scipy.sparse.linalg.norm(C, ord='fro')**2
    else:
        # Dense matrix
        L_A_inv = scipy.linalg.inv(L_A)

        # Multiply by L_B
        if L_B is not None:
            C = L_A_inv @ L_B
        else:
            C = L_A_inv

        trace = numpy.linalg.norm(C, ord='fro')**2

    return trace


# ===========================================
# compute traceinv invert cholesky indirectly
# ===========================================

def compute_traceinv_invert_cholesky_indirectly(
        L_A, L_B, n, sparse, use_cholmod, dtype):
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

    :param L_A: Cholesky factorization of matrix A
    :type L_A: numpy.ndarray, scipy.sprase matrix, or sksparse.cholmod.Factor

    :param L_B: Cholesky factorization of matrix B. If set to None, it is
        assumed that matrix B, and hence L_B, is identify.
    :type L_B: numpy.ndarray, scipy.sprase matrix, or sksparse.cholmod.Factor

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

    # Instead of finding L_A inverse, and then its norm, we directly find norm
    norm2 = 0

    # Cholesky decomposition with scikit-sparse has non-trivial permutation
    # matrix P. Left multiplication by P permutes the rows of matrix. P is a
    # row vector.
    if sparse and use_cholmod and isinstance(L_A, sksparse.cholmod.Factor):
        P = L_A.P()
    else:
        P = None

    # Solve a linear system that finds each of the columns of L_A inverse
    for i in range(n):

        # Handle sparse matrices
        if sparse:

            # Vector e is the i-th column of L_B
            if L_B is not None:
                if P is None:
                    e = L_B[:, i]
                else:
                    e = L_B[P, i]

            else:
                # Assume L_B is identity.
                e = scipy.sparse.lil_matrix((n, 1), dtype=dtype)
                e[i] = 1.0

            # x solves of L_A x = e. Thus, x is the i-th column of L_A inverse.
            if use_cholmod and \
               isinstance(L_A, sksparse.cholmod.Factor):

                # Using cholmod. Note: LDL SHOULD be disabled.
                x = L_A.solve_L(
                        e.tocsc(),
                        use_LDLt_decomposition=False).toarray()

            elif scipy.sparse.isspmatrix(L_A):

                # Using scipy
                x = scipy.sparse.linalg.spsolve_triangular(
                        L_A.tocsr(),
                        e.toarray(),
                        lower=True)

            else:
                raise RuntimeError('Unknown sparse matrix type.')

            # Append to the Frobenius norm of L_A inverse
            norm2 += numpy.sum(x**2)

        else:

            # Vector e is the i-th column of L_B
            if L_B is not None:
                e = L_B[:, i]
            else:
                # Assuming L_B is identity
                e = numpy.zeros(n, dtype=dtype)
                e[i] = 1.0

            # x solves L_A * x = e. Thus, x is the i-th column of L_A inverse
            x = scipy.linalg.solve_triangular(L_A, e, lower=True)

            # Append to the Frobenius norm of L_A inverse
            norm2 += numpy.sum(x**2)

    trace = norm2

    return trace
