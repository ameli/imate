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
import scipy.sparse
from scipy.sparse import isspmatrix
import multiprocessing
from .._linear_algebra.matrix_utilities import get_data_type_name, get_nnz, \
        get_density
from ..__version__ import __version__


# ============
# exact method
# ============

def exact_method(
        A,
        gram=False,
        p=1.0,
        return_info=False):
    """
    Trace of matrix using exact (direct) method.

    Given the matrix :math:`\\mathbf{A}` and the non-negative integer exponent
    :math:`p \\geq 0`, the following is computed:

    .. math::

        \\mathrm{trace} \\left(\\mathbf{A}^p \\right).

    If ``gram`` is `True`, then :math:`\\mathbf{A}` in the above is replaced by
    the Gramian matrix :math:`\\mathbf{A}^{\\intercal} \\mathbf{A}`, and the
    following is instead computed:

    .. math::

        \\mathrm{trace} \\left((\\mathbf{A}^{\\intercal}\\mathbf{A})^p
        \\right).

    Parameters
    ----------

    A : numpy.ndarray, scipy.sparse
        A sparse or dense matrix. If ``gram`` is `True`, the matrix can be
        non-square.

        .. note::

            In the exact method, the matrix cannot be a type of
            :class:`Matrix` or :class:`imate.AffineMatrixFunction` classes.

    gram : bool, default=False
        If `True`, the trace of the Gramian matrix,
        :math:`(\\mathbf{A}^{\\intercal}\\mathbf{A})^p`, is computed. The
        Gramian matrix itself is not directly computed. If `False`, the
        trace of :math:`\\mathbf{A}^p` is computed.

    p : float, default=1.0
        The non-negative integer exponent :math:`p` in :math:`\\mathbf{A}^p`.

    return_info : bool, default=False
        If `True`, this function also returns a dictionary containing
        information about the inner computation, such as process time,
        algorithm settings, etc.

    Returns
    -------

    trace : float or numpy.array
        Trace of matrix.

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
            * ``version``: `str`, version of imate.
            * ``method``: 'exact'

    See Also
    --------

    imate.logdet
    imate.traceinv
    imate.schatten

    Notes
    -----

    With the `exact` method, the trace is computed directly by summing up the
    diagonal elements of the matrix.

    **Computational Complexity:**

    * If :math:`p=1` and ``gram`` is `False`, the computational complexity is
      :math:`\\mathcal{O}(n)`.
      methods.

    * If :math:`p=1` and ``gram`` is `True`, the computational complexity is
      :math:`\\mathcal{O}(n^2)`.

    * If :math:`p=2` and ``gram`` is `False`, the computational complexity is
      :math:`\\mathcal{O}(n^3)`.

    * If :math:`p=2` and ``gram`` is `True`, the computational complexity is
      :math:`\\mathcal{O}(n^3)`.

    * If :math:`p>2`, the computational complexity is
      :math:`\\mathcal{O}(n^3)`.

    .. note::

        When :math:`p=1` and ``gram`` is `False`, the `exact` method should
        always be used. If :math:`p \\neq 1`, use the other methods.

    Examples
    --------

    Compute the trace of a sample sparse Toeplitz matrix created by
    :func:`imate.toeplitz` function.

    .. code-block:: python

        >>> # Import packages
        >>> from imate import toeplitz, trace

        >>> # Generate a sample matrix (a toeplitz matrix)
        >>> A = toeplitz(2, 1, size=100)

        >>> # Compute trace with the exact method (default method)
        >>> trace(A)
        200.0

    Compute the trace of
    :math:`(\\mathbf{A}^{\\intercal} \\mathbf{A})^3`:

    .. code-block:: python

        >>> # Compute trace of the Gramian of A^3 using exact method
        >>> trace(A, p=3, gram=True)
        24307.0

    Print information about the inner computation:

    .. code-block:: python

        >>> tr, info = trace(A, return_info=True)
        >>> print(tr)
        200.0

        >>> # Print dictionary neatly using pprint
        >>> from pprint import pprint
        >>> pprint(info)
        {
            'matrix': {
                'data_type': b'float64',
                'density': 0.0199,
                'exponent': 1.0,
                'gram': False,
                'nnz': 199,
                'num_inquiries': 1,
                'size': (100, 100),
                'sparse': True
            },
            'solver': {
                'method': 'exact',
                'version': '0.14.0'
            },
            'device': {
                'num_cpu_threads': 8,
                'num_gpu_devices': 0,
                'num_gpu_multiprocessors': 0,
                'num_gpu_threads_per_multiprocessor': 0
            },
            'time': {
                'alg_wall_time': 0.00013329205103218555,
                'cpu_proc_time': 0.00017459900000016404,
                'tot_wall_time': 0.00013329205103218555
            }
        }

    **Large matrix:**

    For large matrices, use the `exact` method only if :math:`p = 1` and if
    ``gram`` is `False`.

    .. code-block:: python

        >>> # Generate a matrix of size one million
        >>> A = toeplitz(2, 1, size=1000000)

        >>> # Approximate trace using stochastic Lanczos quadrature
        >>> # with at least 100 Monte-Carlo sampling
        >>> tr, info = trace(A, p=1, gram=False, return_info=True)
        >>> print(tr)
        2000000.0

        >>> # Find the time it took to compute the above
        >>> print(info['time'])
        {
            'tot_wall_time': 0.004113928065635264,
            'alg_wall_time': 0.004113928065635264,
            'cpu_proc_time': 0.0041158319999681225
        }
    """
    # Checking input arguments
    check_arguments(A, gram, p, return_info)

    init_tot_wall_time = time.perf_counter()
    init_cpu_proc_time = time.process_time()

    if p == 0.0:
        trace = numpy.min(A.shape)

    elif p == 1:
        if gram:
            # Compute Frobenius norm
            if scipy.sparse.issparse(A):
                trace = numpy.sum(A.data**2)
            else:
                trace = numpy.linalg.norm(A, ord='fro')**2
        else:
            if scipy.sparse.issparse(A):
                trace = numpy.sum(A.diagonal())
            else:
                trace = numpy.trace(A)

    elif (p % 2 == 0):

        # Since p is an even number, we only need to compute A**p, where
        # p is p // 2.
        if p == 2:

            # Here, p is 1.
            if gram:
                Ap = A.T @ A
            else:
                # Ap (with p=1) is A itself.
                Ap = A
        else:
            # Here, p is 4, 6, or more, and p is half of exponent.
            p = numpy.abs(p) // 2

            # Initialize Ap
            if gram:
                Ap = A.T @ A
                A1 = Ap.copy()
            else:
                Ap = A.copy()
                A1 = A

            # Directly compute power of A by successive matrix multiplication
            for i in range(1, p):
                Ap = Ap @ A1

        if gram:
            if scipy.sparse.issparse(A):
                # Since Ap = (AtA*)**p is symmetric, we compute (Ap).T * Ap
                # instead of (Ap)**2, hence, its Frobenius norm can be used.
                trace = numpy.sum(Ap.data**2)
            else:
                trace = numpy.linalg.norm(Ap, ord='fro')**2
        else:
            if scipy.sparse.issparse(A):
                # Using element-wise matrix multiplication with its transpose.
                # This is fast, compared to direct matrix multiplication.
                trace = numpy.sum(Ap.multiply(Ap.T).data)
            else:
                trace = numpy.sum(numpy.multiply(Ap, Ap.T))

    else:
        # p is an odd number. We need to compute Ap where p is the full
        # exponent
        if gram:
            Ap = A.T @ A
            A1 = Ap.copy()
        else:
            Ap = A.copy()
            A1 = A

        # Directly compute power of A by successive matrix multiplication
        for i in range(1, numpy.abs(p)):
            Ap = Ap @ A1

        if scipy.sparse.issparse(A):
            trace = numpy.sum(Ap.diagonal())
        else:
            trace = numpy.trace(Ap)

    tot_wall_time = time.perf_counter() - init_tot_wall_time
    cpu_proc_time = time.process_time() - init_cpu_proc_time

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
            'method': 'exact'
        }
    }

    if return_info:
        return trace, info
    else:
        return trace


# ===============
# check arguments
# ===============

def check_arguments(A, gram, p, return_info):
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
    elif not isinstance(p, (int, numpy.integer)):
        TypeError('"p" cannot be an integer.')
    elif p < 0:
        ValueError('"p" should be a non-negative integer.')

    # Check return info
    if not isinstance(return_info, bool):
        raise TypeError('"return_info" should be boolean.')
