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
from .._linear_algebra import linear_solver
import multiprocessing

# Cython
from libc.stdlib cimport malloc, free
from cython.parallel cimport parallel, prange
from .._definitions.types cimport IndexType, LongIndexType, DataType
from .._c_basic_algebra cimport cVectorOperations
from .._linear_algebra cimport generate_random_column_vectors

cimport numpy
cimport openmp


# =================
# hutchinson method
# =================

def hutchinson_method(
        A,
        exponent=1,
        assume_matrix='gen',
        min_num_samples=20,
        orthogonalize=True,
        num_threads=0):
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
    check_arguments(A, exponent, assume_matrix, min_num_samples, orthogonalize,
                    num_threads)

    # Parallel processing
    if num_threads < 1:
        num_threads = multiprocessing.cpu_count()

    vector_size = A.shape[0]

    # Allocate random array with Fortran ordering (first index is contiguous)
    # 2D array E should be treated as a matrix, random vectors are columns of E
    E = numpy.empty((vector_size, min_num_samples), dtype=float, order='F')

    # Get c pointer to E
    cdef double[::1, :] memoryview_E = E
    cdef double* cE = &memoryview_E[0, 0]

    init_wall_time = time.perf_counter()
    init_proc_time = time.process_time()

    # Generate orthogonalized random vectors with unit norm
    generate_random_column_vectors[double](cE, vector_size, min_num_samples,
                                           int(orthogonalize), num_threads)

    # In the following, AinvpE is the action of the operator A**(-p) to the
    # vector E. The exponent "p" is the "exponent" argument which is default
    # to one. Ainv means the inverse of A.
    if exponent == 0:
        # Ainvp is the identity matrix
        AinvpE = E

    elif exponent == 1:
        # Perform inv(A) * E. This requires GIL
        AinvpE = linear_solver(A, E, assume_matrix)

    elif exponent > 1:
        # Perform Ainv * Ainv * ... Ainv * E where Ainv is repeated p times
        # where p is the exponent.
        AinvpE = E
        for i in range(exponent):
            AinvpE = linear_solver(A, AinvpE, assume_matrix)

    elif exponent == -1:
        # Performing Ainv**(-1) E, where Ainv**(-1) it A itself.
        AinvpE = A @ E

    elif exponent < -1:
        # Performing Ainv**(-p) E where Ainv**(-p) = A**p.
        AinvpE = E
        for i in range(numpy.abs(exponent)):
            AinvpE = A @ AinvpE

    # To proceed in the following, AinvpE should be in Fortran ordering
    if not AinvpE.flags['F_CONTIGUOUS']:
        AinvpE = numpy.asfortranarray(AinvpE)

    # Get c pointer to AinvpE.
    cdef double[::1, :] memoryview_AinvpE = AinvpE
    cdef double* cAinvpE = &memoryview_AinvpE[0, 0]

    # Stochastic estimator of trace
    cdef double trace = _stochastic_trace_estimator[double](
            cE, cAinvpE, vector_size, min_num_samples, num_threads)

    wall_time = time.perf_counter() - init_wall_time
    proc_time = time.process_time() - init_proc_time

    # Dictionary of output info
    info = {
        'error':
        {
            'absolute_error': None,
            'relative_error': None,
            'error_atol': None,
            'error_rtol': None,
            'confidence_level': None,
            'outlier_significance_level': None
        },
        'convergence':
        {
            'converged': None,
            'all_converged': None,
            'min_min_num_samples': None,
            'max_min_num_samples': None,
            'min_num_samples_used': None,
            'num_outliers': None,
            'samples': None,
            'samples_mean': trace,
            'samples_processed_order': None
        },
        'cpu':
        {
            'wall_time': wall_time,
            'proc_time': proc_time,
            'num_threads': num_threads,
        },
        'solver':
        {
            'orthogonalize': orthogonalize,
            'method': 'hutchinson',
        }
    }

    return trace, info


# ===============
# check arguments
# ===============

def check_arguments(A, exponent, assume_matrix, min_num_samples, orthogonalize,
                    num_threads):
    """
    Checks the type and value of the parameters.
    """

    # Check A
    if (not isinstance(A, numpy.ndarray)) and (not scipy.sparse.issparse(A)):
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
    if not numpy.isscalar(min_num_samples):
        raise TypeError('"min_num_samples" should be a scalar value.')
    elif min_num_samples is None:
        raise ValueError('"min_num_samples" cannot be None.')
    elif not isinstance(min_num_samples, int):
        raise TypeError('"min_num_samples" should be an integer.')
    elif min_num_samples < 1:
        raise ValueError('"min_num_samples" should be at least one.')

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


# ==========================
# stochastic trace estimator
# ==========================

cdef DataType _stochastic_trace_estimator(
        DataType* E,
        DataType* AinvpE,
        const IndexType vector_size,
        const IndexType num_vectors,
        const IndexType num_parallel_threads) nogil:
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

    # Set the number of threads
    openmp.omp_set_num_threads(num_parallel_threads)

    # Initialize openmp lock to setup a critical section
    cdef openmp.omp_lock_t lock
    openmp.omp_init_lock(&lock)

    cdef IndexType i
    cdef DataType* inner_prod = \
        <DataType*> malloc(sizeof(DataType) * num_vectors)
    cdef DataType summand = 0.0

    # Shared-memory parallelism over vectors
    with nogil, parallel():
        for i in prange(num_vectors, schedule='static'):

            # Inner product of i-th column of E and AinvpE (Fortran contiguous)
            inner_prod[i] = cVectorOperations[DataType].inner_product(
                    &E[i*vector_size], &AinvpE[i*vector_size], vector_size)

            # Critical section
            openmp.omp_set_lock(&lock)

            # Sum to compute average later
            summand += inner_prod[i]

            # End of critical section
            openmp.omp_unset_lock(&lock)

    free(inner_prod)

    # Expectation value of Monte-Carlo samples
    cdef DataType trace = vector_size * summand / num_vectors

    return trace
