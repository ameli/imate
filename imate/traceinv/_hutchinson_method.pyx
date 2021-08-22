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

# Python
import time
import numpy
import scipy.sparse
from scipy.sparse import isspmatrix
import multiprocessing
from ..__version__ import __version__
from .._linear_algebra import linear_solver
from ._convergence_tools import check_convergence, average_estimates
from .._trace_estimator.trace_estimator_plot_utilities import plot_convergence
from ._hutchinson_method_utilities import check_arguments, print_summary
from .._linear_algebra.matrix_utilities import get_data_type_name, get_nnz, \
        get_density

# Cython
from .._c_basic_algebra cimport cVectorOperations
from .._linear_algebra cimport generate_random_column_vectors


# =================
# hutchinson method
# =================

def hutchinson_method(
        A,
        exponent=1,
        gram=False,
        assume_matrix='gen',
        min_num_samples=10,
        max_num_samples=50,
        error_atol=None,
        error_rtol=1e-2,
        confidence_level=0.95,
        outlier_significance_level=0.001,
        solver_tol=1e-6,
        orthogonalize=True,
        num_threads=0,
        verbose=False,
        plot=False):
    """
    Computes the trace of inverse of a matrix by Hutchinson method.

    The random vectors have Rademacher distribution. Compared to the Gaussian
    distribution, the Rademacher distribution yields estimation of trace with
    lower variance.

    .. note::

        In the is function, the generated set of random vectors are
        orthogonalized using modified Gram-Schmidt process. Hence, they no
        longer have Rademacher distribution. By orthogonalization, the solution
        seem to have a better convergence.

    :param A: invertible matrix
    :type A: numpy.ndarray

    :param assume_matrix: Assumption about matrix. It can be either ``gen``
        (default) for generic matrix, ``pos`` for positive definite matrix,
        ``sym`` for symmetric matrix, or ``sym_pos`` for symmetric and positive
        definite matrix.
    :type assume_matrix: string

    :param min_num_samples: number of Monte-Carlo random samples
    :type min_num_samples: int

    :param orthogonalize: A flag to indicate whether the set of initial random
        vectors be orthogonalized. If not, the distribution of the initial
        random vectors follows the Rademacher distribution.
    :type orthogonalize: bool

    :return: Trace of matrix ``A``.
    :rtype: float
    """

    # Checking input arguments
    error_atol, error_rtol = check_arguments(
            A, gram, exponent, assume_matrix, min_num_samples, max_num_samples,
            error_atol, error_rtol, confidence_level,
            outlier_significance_level, solver_tol, orthogonalize, num_threads,
            verbose, plot)

    # If the number of random vectors exceed the size of the vectors they
    # cannot be linearly independent and extra calculation with them will be
    # redundant.
    if A.shape[0] < max_num_samples:
        max_num_samples = A.shape[0]
    if A.shape[0] < min_num_samples:
        min_num_samples = A.shape[0]

    # Parallel processing
    if num_threads < 1:
        num_threads = multiprocessing.cpu_count()

    # Dispatch depending on 32-bit or 64-bit
    data_type_name = get_data_type_name(A)
    if data_type_name == b'float32':
        trace, error, num_outliers, samples, processed_samples_indices, \
                num_processed_samples, num_samples_used, converged, \
                tot_wall_time, alg_wall_time, cpu_proc_time = \
                _hutchinson_method_float(A, gram, exponent, assume_matrix,
                                         min_num_samples, max_num_samples,
                                         error_atol, error_rtol,
                                         confidence_level,
                                         outlier_significance_level,
                                         solver_tol, orthogonalize,
                                         num_threads)

    elif data_type_name == b'float64':
        trace, error, num_outliers, samples, processed_samples_indices, \
                num_processed_samples, num_samples_used, converged, \
                tot_wall_time, alg_wall_time, cpu_proc_time = \
                _hutchinson_method_double(A, gram, exponent, assume_matrix,
                                          min_num_samples, max_num_samples,
                                          error_atol, error_rtol,
                                          confidence_level,
                                          outlier_significance_level,
                                          solver_tol, orthogonalize,
                                          num_threads)
    else:
        raise TypeError('Data type should be either "float32" or "float64"')

    # Dictionary of output info
    info = {
        'matrix':
        {
            'data_type': data_type_name,
            'gram': gram,
            'exponent': exponent,
            'assume_matrix': assume_matrix,
            'size': A.shape[0],
            'sparse': isspmatrix(A),
            'nnz': get_nnz(A),
            'density': get_density(A),
            'num_inquiries': 1
        },
        'error':
        {
            'absolute_error': error,
            'relative_error': error / numpy.abs(trace),
            'error_atol': error_atol,
            'error_rtol': error_rtol,
            'confidence_level': confidence_level,
            'outlier_significance_level': outlier_significance_level
        },
        'convergence':
        {
            'converged': bool(converged),
            'min_num_samples': min_num_samples,
            'max_num_samples': max_num_samples,
            'num_samples_used': num_samples_used,
            'num_outliers': num_outliers,
            'samples': samples,
            'samples_mean': trace,
            'samples_processed_order': processed_samples_indices
        },
        'device':
        {
            'num_cpu_threads': num_threads,
            'num_gpu_devices': 0,
            'num_gpu_multiprocessors': 0,
            'num_gpu_threads_per_multiprocessor': 0
        },
        'time':
        {
            'tot_wall_time': tot_wall_time,
            'alg_wall_time': alg_wall_time,
            'cpu_proc_time': cpu_proc_time,
        },
        'solver':
        {
            'version': __version__,
            'orthogonalize': orthogonalize,
            'solver_tol': solver_tol,
            'method': 'hutchinson',
        }
    }

    # print summary
    if verbose:
        print_summary(info)

    # Plot results
    if plot:
        plot_convergence(info)

    return trace, info


# =======================
# hutchinson method float
# =======================

def _hutchinson_method_float(
        A,
        gram,
        exponent,
        assume_matrix,
        min_num_samples,
        max_num_samples,
        error_atol,
        error_rtol,
        confidence_level,
        outlier_significance_level,
        solver_tol,
        orthogonalize,
        num_threads):
    """
    This method processes single precision (32-bit) matrix ``A``.
    """

    vector_size = A.shape[0]

    # Allocate random array with Fortran ordering (first index is contiguous)
    # 2D array E should be treated as a matrix, random vectors are columns of E
    E = numpy.empty((vector_size, max_num_samples), dtype=numpy.float32,
                    order='F')

    # Get c pointer to E
    cdef float[::1, :] memoryview_E = E
    cdef float* cE = &memoryview_E[0, 0]

    init_tot_wall_time = time.perf_counter()
    init_cpu_proc_time = time.process_time()

    # Generate orthogonalized random vectors with unit norm
    generate_random_column_vectors[float](cE, vector_size, max_num_samples,
                                          int(orthogonalize), num_threads)

    samples = numpy.zeros((max_num_samples, ), dtype=numpy.float32)
    processed_samples_indices = numpy.zeros((max_num_samples, ), dtype=int)
    samples[:] = numpy.nan
    cdef int num_processed_samples = 0
    cdef int num_samples_used = 0
    cdef int converged = 0

    init_alg_wall_time = time.perf_counter()

    # Monte-Carlo sampling
    for i in range(max_num_samples):

        if converged == 0:

            # Stochastic estimator of trace using the i-th column of E
            samples[i] = _stochastic_trace_estimator_float(
                    A, E[:, i], gram, exponent, assume_matrix, solver_tol)

            # Store the index of processed samples
            processed_samples_indices[num_processed_samples] = i
            num_processed_samples += 1

            # Check whether convergence criterion has been met to stop.
            # This check can also be done after another parallel thread
            # set all_converged to "1", but we continue to update error.
            converged, num_samples_used = check_convergence(
                    samples, min_num_samples, processed_samples_indices,
                    num_processed_samples, confidence_level, error_atol,
                    error_rtol)

    alg_wall_time = time.perf_counter() - init_alg_wall_time

    trace, error, num_outliers = average_estimates(
            confidence_level, outlier_significance_level, max_num_samples,
            num_samples_used, processed_samples_indices, samples)

    tot_wall_time = time.perf_counter() - init_tot_wall_time
    cpu_proc_time = time.process_time() - init_cpu_proc_time

    return trace, error, num_outliers, samples, processed_samples_indices, \
        num_processed_samples, num_samples_used, converged, tot_wall_time, \
        alg_wall_time, cpu_proc_time


# ========================
# hutchinson method double
# ========================

def _hutchinson_method_double(
        A,
        gram,
        exponent,
        assume_matrix,
        min_num_samples,
        max_num_samples,
        error_atol,
        error_rtol,
        confidence_level,
        outlier_significance_level,
        solver_tol,
        orthogonalize,
        num_threads):
    """
    This method processes double precision (64-bit) matrix ``A``.
    """

    vector_size = A.shape[0]

    # Allocate random array with Fortran ordering (first index is contiguous)
    # 2D array E should be treated as a matrix, random vectors are columns of E
    E = numpy.empty((vector_size, max_num_samples), dtype=numpy.float64,
                    order='F')

    # Get c pointer to E
    cdef double[::1, :] memoryview_E = E
    cdef double* cE = &memoryview_E[0, 0]

    init_tot_wall_time = time.perf_counter()
    init_cpu_proc_time = time.process_time()

    # Generate orthogonalized random vectors with unit norm
    generate_random_column_vectors[double](cE, vector_size, max_num_samples,
                                           int(orthogonalize), num_threads)

    samples = numpy.zeros((max_num_samples, ), dtype=numpy.float64)
    processed_samples_indices = numpy.zeros((max_num_samples, ), dtype=int)
    samples[:] = numpy.nan
    cdef int num_processed_samples = 0
    cdef int num_samples_used = 0
    cdef int converged = 0

    init_alg_wall_time = time.perf_counter()

    # Monte-Carlo sampling
    for i in range(max_num_samples):

        if converged == 0:

            # Stochastic estimator of trace using the i-th column of E
            samples[i] = _stochastic_trace_estimator_double(
                    A, E[:, i], gram, exponent, assume_matrix, solver_tol)

            # Store the index of processed samples
            processed_samples_indices[num_processed_samples] = i
            num_processed_samples += 1

            # Check whether convergence criterion has been met to stop.
            # This check can also be done after another parallel thread
            # set all_converged to "1", but we continue to update error.
            converged, num_samples_used = check_convergence(
                    samples, min_num_samples, processed_samples_indices,
                    num_processed_samples, confidence_level, error_atol,
                    error_rtol)

    alg_wall_time = time.perf_counter() - init_alg_wall_time

    trace, error, num_outliers = average_estimates(
            confidence_level, outlier_significance_level, max_num_samples,
            num_samples_used, processed_samples_indices, samples)

    tot_wall_time = time.perf_counter() - init_tot_wall_time
    cpu_proc_time = time.process_time() - init_cpu_proc_time

    return trace, error, num_outliers, samples, processed_samples_indices, \
        num_processed_samples, num_samples_used, converged, tot_wall_time, \
        alg_wall_time, cpu_proc_time


# ================================
# stochastic trace estimator float
# ================================

cdef float _stochastic_trace_estimator_float(
        A,
        E,
        gram,
        exponent,
        assume_matrix,
        solver_tol) except *:
    """
    Stochastic trace estimator based on set of vectors E and AinvpE.

    :param E: Set of random vectors of shape ``(vector_size, num_vectors)``.
        Note this is Fortran ordering, meaning that the first index is
        contiguous. Hence, to call the i-th vector, use ``&E[0][i]``.
        Here, iteration over the first index is continuous.
    :type E: cython memoryview (float)

    :param AinvpE: Set of random vectors of the same shape as ``E``.
        Note this is Fortran ordering, meaning that the first index is
        contiguous. Hence, to call the i-th vector, use ``&AinvpE[0][i]``.
        Here, iteration over the first index is continuous.
    :type AinvpE: cython memoryview (float)

    :param num_vectors: Number of columns of vectors array.
    :type num_vectors: int

    :param vector_size: Number of rows of vectors array.
    :type vector_size: int

    :param num_parallel_threads: Number of OpenMP parallel threads
    :type num_parallel_threads: int

    :return: Trace estimation.
    :rtype: float
    """

    # In the following, AinvpE is the action of the operator A**(-p) to the
    # vector E. The exponent "p" is the "exponent" argument which is default
    # to one. Ainv means the inverse of A.
    if exponent == 0:
        # Ainvp is the identity matrix
        AinvpE = E

    elif exponent == 1:
        # Perform inv(A) * E. This requires GIL
        if gram:
            AinvpE = linear_solver(A.T, E, assume_matrix, solver_tol)
        else:
            AinvpE = linear_solver(A, E, assume_matrix, solver_tol)

    elif exponent > 1:
        # Perform Ainv * Ainv * ... Ainv * E where Ainv is repeated p times
        # where p is the exponent.
        AinvpE = E

        if gram:
            AtA = A.T @ A
            for i in range(exponent):
                AinvpE = linear_solver(AtA, AinvpE, assume_matrix, solver_tol)
        else:
            for i in range(exponent):
                AinvpE = linear_solver(A, AinvpE, assume_matrix, solver_tol)

    elif exponent == -1:
        # Performing Ainv**(-1) E, where Ainv**(-1) it A itself.
        AinvpE = A @ E

    elif exponent < -1:
        # Performing Ainv**(-p) E where Ainv**(-p) = A**p.
        AinvpE = E
        if gram:
            AtA = A.T @ A
            for i in range(numpy.abs(exponent)):
                AinvpE = AtA @ AinvpE
        else:
            for i in range(numpy.abs(exponent)):
                AinvpE = A @ AinvpE

    # Get c pointer to E
    cdef float[:] memoryview_E = E
    cdef float* cE = &memoryview_E[0]

    # Get c pointer to AinvpE.
    cdef float[:] memoryview_AinvpE = AinvpE
    cdef float* cAinvpE = &memoryview_AinvpE[0]

    # Inner product of E and AinvpE
    cdef int vector_size = A.shape[0]
    cdef float inner_prod

    if gram and (numpy.abs(exponent) == 1):
        inner_prod = cVectorOperations[float].inner_product(cAinvpE, cAinvpE,
                                                            vector_size)
    else:
        inner_prod = cVectorOperations[float].inner_product(cE, cAinvpE,
                                                            vector_size)

    # Hutcinson trace estimate
    cdef float trace_estimate = vector_size * inner_prod

    return trace_estimate


# =================================
# stochastic trace estimator double
# =================================

cdef double _stochastic_trace_estimator_double(
        A,
        E,
        gram,
        exponent,
        assume_matrix,
        solver_tol) except *:
    """
    Stochastic trace estimator based on set of vectors E and AinvpE.

    :param E: Set of random vectors of shape ``(vector_size, num_vectors)``.
        Note this is Fortran ordering, meaning that the first index is
        contiguous. Hence, to call the i-th vector, use ``&E[0][i]``.
        Here, iteration over the first index is continuous.
    :type E: cython memoryview (double)

    :param AinvpE: Set of random vectors of the same shape as ``E``.
        Note this is Fortran ordering, meaning that the first index is
        contiguous. Hence, to call the i-th vector, use ``&AinvpE[0][i]``.
        Here, iteration over the first index is continuous.
    :type AinvpE: cython memoryview (double)

    :param num_vectors: Number of columns of vectors array.
    :type num_vectors: int

    :param vector_size: Number of rows of vectors array.
    :type vector_size: int

    :param num_parallel_threads: Number of OpenMP parallel threads
    :type num_parallel_threads: int

    :return: Trace estimation.
    :rtype: double
    """

    # In the following, AinvpE is the action of the operator A**(-p) to the
    # vector E. The exponent "p" is the "exponent" argument which is default
    # to one. Ainv means the inverse of A.
    if exponent == 0:
        # Ainvp is the identity matrix
        AinvpE = E

    elif exponent == 1:
        # Perform inv(A) * E. This requires GIL
        if gram:
            AinvpE = linear_solver(A.T, E, assume_matrix, solver_tol)
        else:
            AinvpE = linear_solver(A, E, assume_matrix, solver_tol)

    elif exponent > 1:
        # Perform Ainv * Ainv * ... Ainv * E where Ainv is repeated p times
        # where p is the exponent.
        AinvpE = E

        if gram:
            AtA = A.T @ A
            for i in range(exponent):
                AinvpE = linear_solver(AtA, AinvpE, assume_matrix, solver_tol)
        else:
            for i in range(exponent):
                AinvpE = linear_solver(A, AinvpE, assume_matrix, solver_tol)

    elif exponent == -1:
        # Performing Ainv**(-1) E, where Ainv**(-1) it A itself.
        AinvpE = A @ E

    elif exponent < -1:
        # Performing Ainv**(-p) E where Ainv**(-p) = A**p.
        AinvpE = E
        if gram:
            AtA = A.T @ A
            for i in range(numpy.abs(exponent)):
                AinvpE = AtA @ AinvpE
        else:
            for i in range(numpy.abs(exponent)):
                AinvpE = A @ AinvpE

    # Get c pointer to E
    cdef double[:] memoryview_E = E
    cdef double* cE = &memoryview_E[0]

    # Get c pointer to AinvpE.
    cdef double[:] memoryview_AinvpE = AinvpE
    cdef double* cAinvpE = &memoryview_AinvpE[0]

    # Inner product of E and AinvpE
    cdef int vector_size = A.shape[0]
    cdef double inner_prod

    if gram and (numpy.abs(exponent) == 1):
        inner_prod = cVectorOperations[double].inner_product(cAinvpE, cAinvpE,
                                                             vector_size)
    else:
        inner_prod = cVectorOperations[double].inner_product(cE, cAinvpE,
                                                             vector_size)

    # Hutcinson trace estimate
    cdef double trace_estimate = vector_size * inner_prod

    return trace_estimate
