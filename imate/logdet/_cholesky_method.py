# SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the license found in the LICENSE.txt file in the root directory
# of this source tree.


# =======
# Imports
# =======

import time
import numpy
import scipy
import scipy.linalg
from scipy.sparse import isspmatrix
import multiprocessing
from .._linear_algebra.matrix_utilities import get_data_type_name, get_nnz, \
        get_density

try:
    from sksparse.cholmod import cholesky as sk_cholesky
    suitesparse_installed = True
except ImportError:
    suitesparse_installed = False

from .._linear_algebra import sparse_cholesky
from ..__version__ import __version__


# ===============
# cholesky method
# ===============

def cholesky_method(
        A,
        gram=False,
        p=1.0,
        return_info=False,
        cholmod=None):
    """
    Log-determinant of non-singular matrix using Cholesky method.

    Given the matrix :math:`\\mathbf{A}` and the real exponent :math:`p`, the
    following is computed:

    .. math::

        \\mathrm{logdet} \\left(\\mathbf{A}^p \\right) = p \\log_e \\vert
        \\det (\\mathbf{A}) \\vert.

    If ``gram`` is `True`, then :math:`\\mathbf{A}` in the above is replaced by
    the Gramian matrix :math:`\\mathbf{A}^{\\intercal} \\mathbf{A}`. In this
    case, if the matrix :math:`\\mathvf{A}` is square, then the following is
    instead computed:

    .. math::

        \\mathrm{logdet} \\left((\\mathbf{A}^{\\intercal}\\mathbf{A})^p
        \\right) = 2p \\log_e \\vert \\det (\\mathbf{A}) \\vert.

    Parameters
    ----------

    A : numpy.ndarray, scipy.sparse
        A positive-definite sparse or dense matrix.

        .. warning::

            This function does not pre-check whether the input matrix is
            positive-definite.

        .. note::

            In the Cholesky method, the matrix cannot be a type of
            :class:`Matrix` or :class:`imate.AffineMatrixFunction` classes.

    gram : bool, default=False
        If `True`, the log-determinant of the Gramian matrix,
        :math:`(\\mathbf{A}^{\\intercal}\\mathbf{A})^p`, is computed. The
        Gramian matrix itself is not directly computed. If `False`, the
        log-determinant of :math:`\\mathbf{A}^p` is computed.

    p : float, default=1.0
        The real exponent :math:`p` in :math:`\\mathbf{A}^p`.

    return_info : bool, default=False
        If `True`, this function also returns a dictionary containing
        information about the inner computation, such as process time,
        algorithm settings, etc.

    cholmod : bool, default=None
        If set to `True`, it uses the `Cholmod` library from `scikit-sparse`
        package to compute the Cholesky decomposition. If set to `False`, it
        uses `scipy.sparse.cholesky` method. If set to `None`, first, it tries
        to use Cholmod library,  but if Cholmod is not available, then it uses
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
            * ``size``: `(int, int)`, the size of matrix `A`.
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

    **Algorithm:**

    The log-determinant is computed from the Cholesky decomposition
    :math:`\\mathbf{A} = \\mathbf{L} \\mathbf{L}^{\\intercal}` as

    .. math::

        \\log | \\mathbf{A} | =
        2 \\mathrm{trace}( \\log \\mathrm{diag}(\\mathbf{L})).

    The result is exact (no approximation) and could be used as a benchmark to
    test other methods.

    **Computational Complexity:**

    The computational complexity of this method is
    :math:`\\mathcal{O}(\\frac{1}{3}n^3)` for dense matrices.

    For sparse matrices obtained from 1D, 2D, and 3D meshes, the computational
    complexity of this method is respectively :math:`\\mathcal{O}(n)`,
    :math:`\\mathcal{O}(n^{\\frac{3}{2}})`, and :math:`\\mathcal{O}(n^2)` (see
    for instance [1]_ and [2]_).

    **Implementation:**

    This function is essentially a wrapper for the Cholesky function of the
    `scipy` and `scikit-sparse` packages and is primarily used for testing and
    comparison (benchmarking) against the randomized methods that are
    implemented in this package. If ``cholmod`` is set to `True`, this function
    uses the `Suite Sparse
    <https://people.engr.tamu.edu/davis/suitesparse.html>`_ package to compute
    the Cholesky decomposition.

    References
    ----------

    .. [1] George, A. and Ng, E. (1988). *On the Complexity of Sparse QR and LU
           Factorization of Finite-Element Matrices*. SIAM Journal on
           Scientific and Statistical Computing, volume 9, number 5, pp.
           849-861. `doi: 10.1137/0909057 <https://doi.org/10.1137/0909057>`_.

    .. [2] Davis, T. (2006). *Direct Methods for Sparse Linear Systems*. SIAM.
           `doi: 10.1137/1.9780898718881
           <https://epubs.siam.org/doi/book/10.1137/1.9780898718881>`_.

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
                'size': (100, 100),
                'sparse': True
            },
            'solver': {
                'cholmod_used': True,
                'method': 'cholesky',
                'version': '0.13.0'
            },
            'device': {
                'num_cpu_threads': 8,
                'num_gpu_devices': 0,
                'num_gpu_multiprocessors': 0,
                'num_gpu_threads_per_multiprocessor': 0
            },
            'time': {
                'alg_wall_time': 0.0007234140066429973,
                'cpu_proc_time': 0.0009358710000000325,
                'tot_wall_time': 0.0007234140066429973
            }
        }
    """

    # Check input arguments
    square = check_arguments(A, gram, p, return_info, cholmod)

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

    if p == 0:

        # determinant is 1. Logdet is zero
        trace = 0.0

    else:

        # When A is not square and it Gramian is considered, compute the
        # Gramian directly.
        if gram and (not square):
            Ag = A.T @ A
        else:
            Ag = A

        # Compute logdet of A without the exponent
        if sparse:

            # Sparse matrix
            if use_cholmod:
                # Use Suite Sparse
                Factor = sk_cholesky(Ag)
                trace = Factor.logdet()
            else:
                # Use scipy
                diag_L = sparse_cholesky(
                        Ag, diagonal_only=True).astype(numpy.complex128)
                logdet_L = numpy.real(numpy.sum(numpy.log(diag_L)))
                trace = 2.0*logdet_L

        else:

            # Dense matrix. Use scipy
            L = scipy.linalg.cholesky(Ag, lower=True)
            diag_L = numpy.diag(L).astype(numpy.complex128)
            logdet_L = numpy.real(numpy.sum(numpy.log(diag_L)))
            trace = 2.0*logdet_L

        # Taking into account of the exponent p
        trace = trace*p

    # Gramian matrix. Make this adjustment only when matrix is square. For non
    # square gram matrix, we already used A.T @ A
    if gram and square:
        trace = 2.0 * trace

    tot_wall_time = time.perf_counter() - init_tot_wall_time
    cpu_proc_time = time.process_time() - init_cpu_proc_time

    # Dictionary of output info
    info = {
        'matrix':
        {
            'data_type': get_data_type_name(A),
            'gram': gram,
            'exponent': p,
            'size': A.shape,
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

def check_arguments(
        A,
        gram,
        p,
        return_info,
        cholmod):
    """
    Checks the type and value of the parameters.
    """

    # Check A
    if (not isinstance(A, numpy.ndarray)) and (not scipy.sparse.issparse(A)):
        raise TypeError('Input matrix should be either a "numpy.ndarray" or ' +
                        'a "scipy.sparse" matrix.')

    # Check if the matrix is square or not
    if (A.shape[0] != A.shape[1]):
        square = False
    else:
        square = True

    # Check gram
    if gram is None:
        raise TypeError('"gram" cannot be None.')
    elif not numpy.isscalar(gram):
        raise TypeError('"gram" should be a scalar value.')
    elif not isinstance(gram, bool):
        raise TypeError('"gram" should be boolean.')

    # Check non gram should be square
    if (not gram) and (not square):
        raise ValueError('Non Gramian matrix should be square.')

    # Check p
    if p is None:
        raise TypeError('"p" cannot be None.')
    elif not numpy.isscalar(p):
        raise TypeError('"p" should be a scalar value.')
    elif isinstance(p, complex):
        TypeError('"p" cannot be an integer or a float number.')

    # Check return info
    if not isinstance(return_info, bool):
        raise TypeError('"return_info" should be boolean.')

    # Check cholmod
    if cholmod is not None:
        if not isinstance(cholmod, bool):
            raise TypeError('"cholmod" should be either "None", or boolean.')
        elif cholmod is True and suitesparse_installed is False:
            print(cholmod)
            print(suitesparse_installed)
            raise RuntimeError('"cholmod" method is not available. Either ' +
                               'install "scikit-sparse" package, or set ' +
                               '"cholmod" to "False" or "None".')

    return square
