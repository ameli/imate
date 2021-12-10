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
        exponent=1.0):
    """
    """
    # Checking input arguments
    check_arguments(A, gram, exponent)

    init_tot_wall_time = time.perf_counter()
    init_cpu_proc_time = time.process_time()

    if exponent == 0.0:
        trace = numpy.min(A.shape)

    elif exponent == 1:
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

    elif (exponent % 2 == 0):

        # Since exponent is an even number, we only need to compute A**p, where
        # p is exponent // 2.
        if exponent == 2:

            # Here, p is 1.
            if gram:
                Ap = A.T @ A
            else:
                # Ap (with p=1) is A itself.
                Ap = A
        else:
            # Here, exponent is 4, 6, or more, and p is half of exponent.
            p = numpy.abs(exponent) // 2

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
        # Exponent is an odd number. We need to compute Ap where p is the full
        # exponent.
        if gram:
            Ap = A.T @ A
            A1 = Ap.copy()
        else:
            Ap = A.copy()
            A1 = A

        # Directly compute power of A by successive matrix multiplication
        for i in range(1, numpy.abs(exponent)):
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
            'exponent': exponent,
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
            'method': 'exact'
        }
    }

    return trace, info


# ===============
# check arguments
# ===============

def check_arguments(A, gram, exponent):
    """
    Checks the type and value of the parameters.
    """

    # Check A
    if (not isinstance(A, numpy.ndarray)) and (not scipy.sparse.issparse(A)):
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
    elif exponent < 0:
        ValueError('"exponent" should be a non-negative integer.')
