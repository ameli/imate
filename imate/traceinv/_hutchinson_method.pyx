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
from .._linear_algebra import linear_solver
from ._convergence_tools import check_convergence, average_estimates
from .._trace_estimator.trace_estimator_plot_utilities import plot_convergence
from ._utilities import get_data_type_name, get_nnz, get_density, print_summary

# Cython
from .._c_basic_algebra cimport cVectorOperations
from .._linear_algebra cimport generate_random_column_vectors


# =================
# hutchinson method
# =================

def hutchinson_method(
        A,
        exponent=1,
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
            A, exponent, assume_matrix, min_num_samples, max_num_samples,
            error_atol, error_rtol, confidence_level,
            outlier_significance_level, solver_tol, orthogonalize, num_threads,
            verbose, plot)

    # Parallel processing
    if num_threads < 1:
        num_threads = multiprocessing.cpu_count()

    vector_size = A.shape[0]

    # Allocate random array with Fortran ordering (first index is contiguous)
    # 2D array E should be treated as a matrix, random vectors are columns of E
    E = numpy.empty((vector_size, max_num_samples), dtype=float, order='F')

    # Get c pointer to E
    cdef double[::1, :] memoryview_E = E
    cdef double* cE = &memoryview_E[0, 0]

    init_tot_wall_time = time.perf_counter()
    init_cpu_proc_time = time.process_time()

    # Generate orthogonalized random vectors with unit norm
    generate_random_column_vectors[double](cE, vector_size, max_num_samples,
                                           int(orthogonalize), num_threads)

    samples = numpy.zeros((max_num_samples, ), dtype=float)
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
            samples[i] = _stochastic_trace_estimator(A, E[:, i], exponent,
                                                     assume_matrix, solver_tol)

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

    # Dictionary of output info
    info = {
        'matrix':
        {
            'data_type': get_data_type_name(A),
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


# ===============
# check arguments
# ===============

def check_arguments(
        A,
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
        num_threads,
        verbose,
        plot):
    """
    Checks the type and value of the parameters.
    """

    # Check A
    if (not isinstance(A, numpy.ndarray)) and (not isspmatrix(A)):
        raise TypeError('Input matrix should be either a "numpy.ndarray" or ' +
                        'a "scipy.sparse" matrix.')
    elif A.shape[0] != A.shape[1]:
        raise ValueError('Input matrix should be a square matrix.')

    # Check exponent
    if exponent is None:
        raise TypeError('"exponent" cannot be None.')
    elif not numpy.isscalar(exponent):
        raise TypeError('"exponent" should be a scalar value.')
    elif not isinstance(exponent, int):
        TypeError('"exponent" cannot be an integer.')

    # Check assume_matrix
    if assume_matrix is None:
        raise ValueError('"assume_matrix" cannot be None.')
    elif not isinstance(assume_matrix, basestring):
        raise TypeError('"assume_matrix" must be a string.')
    elif assume_matrix != 'gen' and assume_matrix != "pos" and \
            assume_matrix != "sym" and assume_matrix != "sym_pos":
        raise ValueError('"assume_matrix" should be either "gen", "pos", ' +
                         '"sym, or "sym_pos".')

    # Check min_num_samples
    if min_num_samples is None:
        raise ValueError('"min_num_samples" cannot be None.')
    elif not numpy.isscalar(min_num_samples):
        raise TypeError('"min_num_samples" should be a scalar value.')
    elif not isinstance(min_num_samples, int):
        raise TypeError('"min_num_samples" should be an integer.')
    elif min_num_samples < 1:
        raise ValueError('"min_num_samples" should be at least one.')

    # Check max_num_samples
    if max_num_samples is None:
        raise ValueError('"max_num_samples" cannot be None.')
    elif not numpy.isscalar(max_num_samples):
        raise TypeError('"max_num_samples" should be a scalar value.')
    elif not isinstance(max_num_samples, int):
        raise TypeError('"max_num_samples" should be an integer.')
    elif max_num_samples < 1:
        raise ValueError('"max_num_samples" should be at least one.')

    # Check min and max num samples
    if min_num_samples > max_num_samples:
        raise ValueError('"min_num_samples" cannot be greater than ' +
                         '"max_num_samples".')

    # Check convergence absolute tolerance
    if error_atol is None:
        error_atol = 0.0
    elif not numpy.isscalar(error_atol):
        raise TypeError('"error_atol" should be a scalar value.')
    elif not isinstance(error_atol, (int, float)):
        raise TypeError('"error_atol" should be a float number.')
    elif error_atol < 0.0:
        raise ValueError('"error_atol" cannot be negative.')

    # Check error relative tolerance
    if error_rtol is None:
        error_rtol = 0.0
    elif not numpy.isscalar(error_rtol):
        raise TypeError('"error_rtol" should be a scalar value.')
    elif not isinstance(error_rtol, (int, float)):
        raise TypeError('"error_rtol" should be a float number.')
    elif error_rtol < 0.0:
        raise ValueError('"error_rtol" cannot be negative.')

    # Check confidence level
    if confidence_level is None:
        raise TypeError('"confidence_level" cannot be None.')
    elif not numpy.isscalar(confidence_level):
        raise TypeError('"confidence_level" should be a scalar.')
    elif not isinstance(confidence_level, (int, float)):
        raise TypeError('"confidence_level" should be a float number.')
    elif confidence_level < 0.0 or confidence_level > 1.0:
        raise ValueError('"confidence_level" should be between 0 and 1.')

    # Check outlier significance level
    if outlier_significance_level is None:
        raise TypeError('"outlier_significance_level" cannot be None.')
    elif not numpy.isscalar(outlier_significance_level):
        raise TypeError('"outlier_significance_level" should be a scalar.')
    elif not isinstance(outlier_significance_level, (int, float)):
        raise TypeError('"outlier_significance_level" must be a float number.')
    elif outlier_significance_level < 0.0 or outlier_significance_level > 1.0:
        raise ValueError(
                '"outlier_significance_level" must be in [0, 1] interval.')

    # Compare outlier significance level and confidence level
    if outlier_significance_level > 1.0 - confidence_level:
        raise ValueError('The sum of "confidence_level" and ' +
                         '"outlier_significance_level" should be less than 1.')

    # Check solver tol
    if solver_tol is not None and not numpy.isscalar(solver_tol):
        raise TypeError('"solver_tol" should be a scalar value.')
    elif solver_tol is not None and not isinstance(solver_tol, (int, float)):
        raise TypeError('"solver_tol" should be a float number.')
    elif solver_tol is not None and solver_tol < 0.0:
        raise ValueError('"lancozs_tol" cannot be negative.')

    # Check orthogonalize
    if orthogonalize is None:
        raise TypeError('"orthogonalize" cannot be None.')
    elif not numpy.isscalar(orthogonalize):
        raise TypeError('"orthogonalize" should be a scalar value.')
    elif not isinstance(orthogonalize, bool):
        raise TypeError('"orthogonalize" should be boolean.')

    # Check num_threads
    if num_threads is None:
        raise TypeError('"num_threads" cannot be None.')
    elif not numpy.isscalar(num_threads):
        raise TypeError('"num_threads" should be a scalar value.')
    elif not isinstance(num_threads, int):
        raise TypeError('"num_threads" should be an integer.')
    elif num_threads < 0:
        raise ValueError('"num_threads" should be a non-negative integer.')

    # Check verbose
    if verbose is None:
        raise TypeError('"verbose" cannot be None.')
    elif not numpy.isscalar(verbose):
        raise TypeError('"verbose" should be a scalar value.')
    elif not isinstance(verbose, bool):
        raise TypeError('"verbose" should be boolean.')

    # Check plot
    if plot is None:
        raise TypeError('"plot" cannot be None.')
    elif not numpy.isscalar(plot):
        raise TypeError('"plot" should be a scalar value.')
    elif not isinstance(plot, bool):
        raise TypeError('"plot" should be boolean.')

    # Check if plot modules exist
    if plot is True:
        try:
            from .._utilities.plot_utilities import matplotlib      # noqa F401
            from .._utilities.plot_utilities import load_plot_settings
            load_plot_settings()
        except ImportError:
            raise ImportError('Cannot import modules for plotting. Either ' +
                              'install "matplotlib" and "seaborn" packages, ' +
                              'or set "plot=False".')

    return error_atol, error_rtol


# ==========================
# stochastic trace estimator
# ==========================

cdef double _stochastic_trace_estimator(
        A,
        E,
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
        AinvpE = linear_solver(A, E, assume_matrix, solver_tol)

    elif exponent > 1:
        # Perform Ainv * Ainv * ... Ainv * E where Ainv is repeated p times
        # where p is the exponent.
        AinvpE = E
        for i in range(exponent):
            AinvpE = linear_solver(A, AinvpE, assume_matrix, solver_tol)

    elif exponent == -1:
        # Performing Ainv**(-1) E, where Ainv**(-1) it A itself.
        AinvpE = A @ E

    elif exponent < -1:
        # Performing Ainv**(-p) E where Ainv**(-p) = A**p.
        AinvpE = E
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
    cdef double inner_prod = cVectorOperations[double].inner_product(
                    cE, cAinvpE, vector_size)

    # Hutcinson trace estimate
    cdef double trace_estimate = vector_size * inner_prod

    return trace_estimate
