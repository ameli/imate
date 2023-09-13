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
import numpy.linalg
import scipy
import scipy.linalg
import scipy.sparse
import scipy.sparse.linalg
from scipy.sparse import isspmatrix
import multiprocessing
from .._linear_algebra.matrix_utilities import get_data_type_name, get_nnz, \
        get_density
from ..__version__ import __version__


# =================
# eigenvalue method
# =================

def eigenvalue_method(
        A,
        gram=False,
        p=1.0,
        return_info=False,
        eigenvalues=None,
        assume_matrix='gen',
        non_zero_eig_fraction=0.9):
    """
    Log-determinant of non-singular matrix using eigenvalue method.

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
        A non-singular sparse or dense matrix. If ``gram`` is `True`, the
        input matrix can be non-square.

        .. note::

            In the eigenvalue method, the matrix cannot be a type of
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

    eigenvalues : array_like [`float`], default=None
        The array of all eigenvalues of `A`, if available. The size of this
        array must be the same as the size of `A`. If `None`, the eigenvalues
        of `A` are computed.

    assume_matrix : str {'gen', 'sym'}, default: 'gen'
        Type of matrix. `gen` assumes generic matrix, while `sym` assumes
        `A` is symmetric.

    non_zero_eig_fraction : float, default=0.9
        A fraction (between `0` and `1`) of eigenvalues assumed to be non-zero.
        For large matrices, it is not possible to compute all eigenvalues, and
        only the largest eigenvalues can be computed and the rest are assumed
        to be negligible. By setting this parameter, a fraction of
        non-negligible eigenvalues is determined.

    Returns
    -------

    logdet : float or numpy.array
        Log-determinant of `A`.

    info : dict
        (Only if ``return_info`` is `True`) A dictionary of information with
        the following keys.

        * ``matrix``:
            * ``data_type``: `str`, {`float32`, `float64`, `float128`}, type of
              the matrix data.
            * ``gram``: `bool`, whether the matrix `A` or its Gramian is
              considered.
            * ``exponent``: `float`, the exponent `p` in :math:`\\mathbf{A}^p`.
            * ``assume_matrix``: `str`, {`gen`, `sym`}, determines whether
              matrix is generic or symmetric.
            * ``size``: `(int, int)`, the size of matrix `A`.
            * ``sparse``: `bool`, whether the matrix `A` is sparse or dense.
            * ``nnz``: `int`, if `A` is sparse, the number of non-zero elements
              of `A`.
            * ``density``: `float`, if `A` is sparse, the density of `A`, which
              is the `nnz` divided by size squared.
            * ``num_inquiries``: `int`, for the `eigenvalue` method, this is
              always `1`.

        * ``device``:
            * ``num_cpu_threads``: `int`, number of CPU threads used in shared
              memory parallel processing.
            * ``num_gpu_devices``: `int`, for the `eigenvalue` method, this is
              always `0`.
            * ``num_gpu_multiprocessors``: `int`, for the `eigenvalue` method,
              this is always `0`.
            * ``num_gpu_threads_per_multiprocessor``: `int`, for `eigenvalue`
              method, this is always `0`.

        * ``time``:
            * ``tot_wall_time``: `float`, total elapsed time of computation.
            * ``alg_wall_time``: `float`, elapsed time of computation during
              only the algorithm execution.
            * ``cpu_proc_time``: `float`, the CPU processing time of
              computation.

        * ``solver``:
            * ``version``: `str`, version of imate.
            * ``method``: 'eigenvalue'

    See Also
    --------

    imate.trace
    imate.traceinv
    imate.schatten

    Notes
    -----

    **Computational Complexity:**

    The eigenvalue method uses spectral decomposition. The computational
    complexity of this method is :math:`\\mathcal{O}(n^3)` where :math:`n` is
    the matrix size. This method is only suitable for small matrices
    (:math:`n < 2^{12}`). The solution is exact and can be used as a benchmark
    to test randomized methods of computing log-determinant.

    .. warning::

        It is not recommended to use this method for sparse matrices, as not
        all eigenvalues of sparse matrices can be computed.

    Examples
    --------

    **Dense matrix:**

    Compute the log-determinant of a sample sparse Toeplitz matrix created
    by :func:`imate.toeplitz` function:

    .. code-block:: python

        >>> # Import packages
        >>> from imate import toeplitz, logdet

        >>> # Generate a sample symmetric matrix (a toeplitz matrix)
        >>> A = toeplitz(2, 1, size=100, gram=True)

        >>> # Convert the sparse matrix to a dense matrix
        >>> A = A.toarray()

        >>> # Compute log-determinant with Cholesky method (default method)
        >>> logdet(A, method='eigenvalue', assume_matrix='sym')
        138.62943611198907

    **Precomputed Eigenvalues:**

    Alternatively, compute the eigenvalues of `A` in advance, and pass it to
    this function:

    .. code-block:: python
        :emphasize-lines: 6

        >>> # Compute eigenvalues of symmetric matrix A.
        >>> from scipy.linalg import eigh
        >>> eigenvalues = eigh(A, eigvals_only=True)

        >>> # Pass the eigenvalues to logdet function
        >>> logdet(A, method='eigenvalue', eigenvalues=eigenvalues)
        138.62943611198907

    Pre-computing eigenvalues can be useful if :func:`imate.logdet` function
    should be called repeatedly for the same matrix `A` but other parameters
    may change, such as `p`.

    **Print Information:**

    Print information about the inner computation:

    .. code-block:: python

        >>> ld, info = logdet(A, method='eigenvalue', return_info=True)
        >>> print(ld)
        138.6294361119891

        >>> # Print dictionary neatly using pprint
        >>> from pprint import pprint
        >>> pprint(info)
        {
            'matrix': {
                'assume_matrix': 'gen',
                'data_type': b'float64',
                'density': 1.0,
                'exponent': 1.0,
                'gram': False,
                'nnz': 10000,
                'num_inquiries': 1,
                'size': (100, 100),
                'sparse': False
            },
            'solver': {
                'method': 'eigenvalue',
                'version': '0.13.0'
            },
            'device': {
                'num_cpu_threads': 8,
                'num_gpu_devices': 0,
                'num_gpu_multiprocessors': 0,
                'num_gpu_threads_per_multiprocessor': 0
            },
            'time': {
                'alg_wall_time': 0.007327683997573331,
                'cpu_proc_time': 0.014451992999999774,
                'tot_wall_time': 0.007327683997573331
            }
        }

    **Sparse matrix:**

    Using a large matrix and ignoring 10% of its eigenvalues:

    .. code-block:: python

        >>> # Generate a symmetric sparse matrix
        >>> A = toeplitz(2, 1, size=2000, gram=True)

        >>> # Assume only 80% of eigenvalues of A are non-zero
        >>> logdet(A, method='eigenvalue', assume_matrix='sym',
        ...        non_zero_eig_fraction=0.9)
        2451.2640192906174

    The above result is only an approximation since not all eigenvalues of `A`
    are taken into account. To compare with the exact solution, use
    :func:`imate.sample_matrices.toeplitz_logdet` function.

    .. code-block:: python

        >>> from imate.sample_matrices import toeplitz_logdet
        >>> toeplitz_logdet(2, 1, size=2000, gram=True)
        2772.588722239781

    There is a significant difference between the approximation with 90% of
    eigenvalues and the actual solution. Because of this, it is not recommended
    to use the eigenvalue method to compute the log-determinant.
    """

    # Checking input arguments
    square = check_arguments(A, eigenvalues, gram, p, return_info,
                             assume_matrix, non_zero_eig_fraction)

    init_tot_wall_time = time.perf_counter()
    init_cpu_proc_time = time.process_time()

    if eigenvalues is None:

        if square:
            # Matrix is square, either gram or not gram. For the case of gram,
            # we will multiply the logdet by two later.
            eigenvalues = compute_eigenvalues(A, assume_matrix,
                                              non_zero_eig_fraction)

        else:
            # Matrix is gram but A is not square. Compute singular values
            # instead.
            singularvalues = compute_singularvalues(A, non_zero_eig_fraction)

            # Compute eigenvalues of gram matrix from the singular values. Note
            # in this case, there is no need to multiply the logdet by two
            # later.
            eigenvalues = singularvalues ** 2

    # Compute logdet of matrix
    not_nan = numpy.logical_not(numpy.isnan(eigenvalues))
    eig_Ap = eigenvalues[not_nan]**p
    logdet_ = numpy.sum(numpy.log(eig_Ap.astype(numpy.complex128)))

    # Return only the real part
    imag_atol = 1e-8
    imag_rtol = 1e-8
    if isinstance(logdet_, numpy.complex128):

        # Get imaginary part as mod of pi
        imag = numpy.mod(logdet_.imag, numpy.pi)
        if (numpy.abs(imag) < imag_atol) or \
                (numpy.abs(imag - numpy.pi) < imag_atol):
            imag = 0
        if imag > imag_rtol * A.shape[0]:
            raise RuntimeError(
                    'Determinant is not a purely real number. ' +
                    'Real part: %f, Imaginary part: %f' % (logdet_, imag))
        else:
            quotient = int(numpy.abs(logdet_.imag) / numpy.pi + 0.5)

            # For odd quotient, logdet has i*\pi
            if (quotient // 2) == int(quotient / 2.0 + 0.5):
                logdet_ = logdet_.real
            else:
                logdet_ = logdet_.real + complex(0.0, numpy.pi)

    # Gramian matrix. Make this adjustment only when matrix is square. For non
    # square gram matrix, we already used the square of singularvalues.
    if gram and square:
        logdet_ = 2.0 * logdet_

    tot_wall_time = time.perf_counter() - init_tot_wall_time
    cpu_proc_time = time.process_time() - init_cpu_proc_time

    # Dictionary of output info
    info = {
        'matrix':
        {
            'data_type': get_data_type_name(A),
            'gram': gram,
            'exponent': p,
            'assume_matrix': assume_matrix,
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
            'method': 'eigenvalue',
        }
    }

    if return_info:
        return logdet_, info
    else:
        return logdet_


# ===============
# check arguments
# ===============

def check_arguments(
        A,
        eigenvalues,
        gram,
        p,
        return_info,
        assume_matrix,
        non_zero_eig_fraction):
    """
    Checks the type and value of the parameters.
    """

    # Check A
    if (not isinstance(A, numpy.ndarray)) and (not scipy.sparse.issparse(A)):
        raise TypeError('Input matrix should be either a "numpy.ndarray" or ' +
                        'a "scipy.sparse" matrix.')

    # Check eigenvalues
    if eigenvalues is not None:
        if not isinstance(eigenvalues, numpy.ndarray):
            raise TypeError('"eigenvalues" should be a numpy.ndarray.')
        if eigenvalues.size != A.shape[0]:
            raise ValueError('The length of "eigenvalues" does not match ' +
                             'the size of matrix "A".')

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
    elif not isinstance(p, numpy.integer):
        TypeError('"p" cannot be an integer.')

    # Check return info
    if not isinstance(return_info, bool):
        raise TypeError('"return_info" should be boolean.')

    # Check assume_matrix
    if assume_matrix is None:
        raise TypeError('"assume_matrix" cannot be None.')
    elif not isinstance(assume_matrix, str):
        raise TypeError('"assume_matrix" should be a string.')
    elif assume_matrix not in ['gen', 'sym']:
        raise ValueError('"assume_matrix" should be either "gen" or "sym".')

    # Check non_zero_eig_fraction
    if non_zero_eig_fraction is None:
        raise TypeError('"non_zero_eig_fraction" cannot be None.')
    elif not numpy.isscalar(non_zero_eig_fraction):
        raise TypeError('"non_zero_eig_fraction" should be a scalar value.')
    elif not isinstance(non_zero_eig_fraction, float):
        raise TypeError('"non_zero_eig_fraction" should be a float type.')
    elif non_zero_eig_fraction <= 0 or non_zero_eig_fraction >= 1.0:
        raise ValueError('"non_zero_eig_fraction" should be greater then 0.0' +
                         'and smaller than 1.0.')

    return square


# ===================
# compute eigenvalues
# ===================

def compute_eigenvalues(
        A,
        assume_matrix,
        non_zero_eig_fraction,
        tol=1e-4):
    """
    """

    if scipy.sparse.isspmatrix(A):

        # Sparse matrix
        n = A.shape[0]
        eigenvalues = numpy.empty(n)
        eigenvalues[:] = numpy.nan

        # find 90% of eigenvalues, assume the rest are very close to zero.
        num_none_zero_eig = int(n*non_zero_eig_fraction)

        if assume_matrix == 'sym':

            # The logarithm function on the eigenvalues (for computing logdet)
            # takes the greatest effect from both the smallest and largest
            # eigenvalues. The logarithm of the smallest eigenvalues is a large
            # negative value, and the logarithm of the largest eigenvalues are
            # positive large values. ``BE`` option in scipy.sparse.linalg.eigh
            # computes half of eigenvalues from each end of the spectrum, which
            # is what exactly we need here.
            which_eigenvalues = 'BE'

            eigenvalues[:num_none_zero_eig] = \
                scipy.sparse.linalg.eigsh(A, num_none_zero_eig,
                                          which=which_eigenvalues,
                                          return_eigenvectors=False,
                                          tol=tol)
        else:
            # Computing half of eigenvalues from the largest magnitude
            eigenvalues_large = scipy.sparse.linalg.eigs(
                    A, num_none_zero_eig//2, which='LM',
                    return_eigenvectors=False, tol=tol)

            # Computing half of eigenvalues from the smallest magnitude
            eigenvalues_small = scipy.sparse.linalg.eigs(
                    A, num_none_zero_eig//2, which='SM',
                    return_eigenvectors=False, tol=tol)

            # Combine large and small eigenvalues into one array
            num_eigenvalues = eigenvalues_large.size + \
                eigenvalues_small.size
            eigenvalues[:num_eigenvalues] = numpy.r_[eigenvalues_large,
                                                     eigenvalues_small]
    else:

        # Dense matrix
        if assume_matrix == 'sym':
            eigenvalues = scipy.linalg.eigh(A, check_finite=False,
                                            eigvals_only=True)
        else:
            eigenvalues = scipy.linalg.eig(A, check_finite=False)[0]

    return eigenvalues


# ======================
# compute singularvalues
# ======================

def compute_singularvalues(
        A,
        non_zero_eig_fraction,
        tol=1e-4):
    """
    """

    if scipy.sparse.isspmatrix(A):

        # Sparse matrix
        n = numpy.min([A.shape[0], A.shape[1]])
        singularvalues = numpy.empty(n)
        singularvalues[:] = numpy.nan

        # find 90% of singularvalues, assume the rest are very close to zero.
        num_none_zero_eig = int(n*non_zero_eig_fraction)

        # Computing half of singularvalues from the largest magnitude
        singularvalues_large = scipy.sparse.linalg.svds(
                A, num_none_zero_eig//2, which='LM',
                return_singular_vectors=False, tol=tol)

        # Computing half of singularvalues from the smallest magnitude
        singularvalues_small = scipy.sparse.linalg.svds(
                A, num_none_zero_eig//2, which='SM',
                return_singular_vectors=False, tol=tol)

        # Combine large and small singularvalues into one array
        num_singularvalues = singularvalues_large.size + \
            singularvalues_small.size
        singularvalues[:num_singularvalues] = numpy.r_[singularvalues_large,
                                                       singularvalues_small]
    else:

        # Dense matrix
        singularvalues = scipy.linalg.svdvals(A, check_finite=False)

    return singularvalues
