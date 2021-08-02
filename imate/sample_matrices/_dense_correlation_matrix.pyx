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
import numpy
import multiprocessing

# Cython
from cython.parallel cimport parallel, prange
from ._kernels cimport get_kernel, euclidean_distance, _exponential_kernel
from libc.stdlib cimport exit, malloc, free
from libc.stdio cimport printf
from libc.math cimport NAN
from .._definitions.types cimport DataType, kernel_type
cimport cython
cimport openmp

__all__ = ['dense_correlation_matrix']

# To avoid a bug where cython does not recognize long doubler as a type in the
# template functions, we define long_double as an alias
ctypedef long double long_double


# ===============
# generate matrix
# ===============

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _generate_matrix(
        const double[:, ::1] coords,
        const int matrix_size,
        const int dimension,
        const double distance_scale,
        const kernel_type kernel_function,
        const double kernel_param,
        const int num_threads,
        const int verbose,
        DataType* c_correlation_matrix) nogil:
    """
    Generates a dense correlation matrix.

    :param coords: A 2D array containing the coordinates of the spatial set of
        points in the unit hypercube. The first index of this array is the
        point ids and the second index is the dimension of the coordinates.
    :type coords: cython memoryview (double)

    :param matrix_size: The shape of the first index of ``coords``, which is
        also the size of the generated output matrix.
    :type matrix_size: int

    :param dimension: The shape of the second index of ``coords`` array, which
        is the dimension of the spatial points.
    :type dimension: int

    :param distance_scale: A parameter of the correlation function that
        scales distances.
    :type distance_scale: double

    :param nu: The parameter :math:`\\nu` of Matern correlation kernel.
    :type nu: float

    :param num_threads: Number of parallel threads in openmp.
    :type num_threads: int

    :param correlation_matrix: Output array. Correlation matrix of the size
        ``matrix_size``.
    :type correlation_matrix: cython memoryview (double)
    """

    cdef int i, j
    cdef int dim
    cdef int[1] counter
    cdef int percent_update
    cdef int progress

    # percent_update determines how often a progress is being printed
    if matrix_size <= 50:
        percent_update = 20
    elif matrix_size <= 1000:
        percent_update = 10
    elif matrix_size <= 10000:
        percent_update = 5
    elif matrix_size <= 50000:
        percent_update = 2
    else:
        percent_update = 1

    # Set number of parallel threads
    openmp.omp_set_num_threads(num_threads)

    # Initialize openmp lock to setup a critical section
    cdef openmp.omp_lock_t lock_counter
    openmp.omp_init_lock(&lock_counter)

    # Using max possible chunk size for parallel threads
    cdef int chunk_size = int((<double> matrix_size) / num_threads)
    if chunk_size < 1:
        chunk_size = 1

    # Iterate over rows of correlation matrix
    counter[0] = 0
    with nogil, parallel():
        for i in prange(matrix_size, schedule='static', chunksize=chunk_size):
            for j in range(i, matrix_size):

                # Compute correlation
                c_correlation_matrix[i*matrix_size + j] = \
                    <DataType> kernel_function(
                        euclidean_distance(
                            coords[i][:],
                            coords[j][:],
                            distance_scale,
                            dimension),
                        kernel_param)

                # Use symmetry of the correlation matrix
                if i != j:
                    c_correlation_matrix[j*matrix_size + i] = \
                        c_correlation_matrix[i*matrix_size + j]

            # Critical section
            openmp.omp_set_lock(&lock_counter)

            # Update counter
            counter[0] = counter[0] + 1

            # Print progress on every percent_update
            if verbose and (matrix_size * percent_update >= 100):
                if (counter[0] % (matrix_size*percent_update//100) == 0):
                    progress = percent_update * counter[0] // \
                        (matrix_size*percent_update//100)
                    printf('Generate matrix progress: %3d%%\n', progress)

            # Release lock to end the openmp critical section
            openmp.omp_unset_lock(&lock_counter)


# ========================
# dense correlation matrix
# ========================

def dense_correlation_matrix(
        coords,
        distance_scale=0.1,
        kernel='exponential',
        kernel_param=None,
        dtype=r'float64',
        verbose=False):
    """
    Generates a dense correlation matrix.

    .. note::

        If the ``kernel_threshold`` is large, it causes:

            * The correlation matrix :math:`\\mathbf{K}` will not be
              positive-definite.
            * The function :math:`\\mathrm{trace}\\left((\\mathbf{K}+t
              \\mathbf{I})^{-1}\\right)` produces unwanted oscillations.

    :param coords: 2D array of the coordinates of the set of points. The first
        index of the array is the point Ids and its size determines the size of
        the correlation matrix. The second index of the array corresponds to
        the dimension of the spatial points.
    :type coords: numpy.ndarray

    :param distance_scale: A parameter of correlation function that scales
        distance.
    :type distance_scale: float

    :param verbose: If ``True``, prints some information during the process.
    :type verbose: bool

    :return: Correlation matrix. If ``coords`` is ``n*m`` array, the
        correlation matrix has ``n*n`` shape.
    :rtype: numpy.ndarray

    :param nu: The parameter :math:`\\nu` of Matern correlation kernel.
    :type nu: float
    """

    # Makes kernel paramerter a C-type NAN
    if kernel_param is None:
        kernel_param = NAN

    # size of data and the correlation matrix
    matrix_size = coords.shape[0]
    dimension = coords.shape[1]

    # Get number of CPU threads
    num_threads = multiprocessing.cpu_count()

    # Initialize matrix
    correlation_matrix = numpy.zeros((matrix_size, matrix_size), dtype=dtype,
                                     order='C')

    # Memory view of the correlation matrix
    cdef float[:, ::1] mv_correlation_matrix_float
    cdef double[:, ::1] mv_correlation_matrix_double
    cdef long double[:, ::1] mv_correlation_matrix_long_double

    # C pointer to the correlation matrix
    cdef float* c_correlation_matrix_float
    cdef double* c_correlation_matrix_double
    cdef long double* c_correlation_matrix_long_double

    # Get the kernel functon
    cdef kernel_type kernel_function = get_kernel(kernel)

    if dtype == r'float32':

        # Get pointer to the correlation matrix
        mv_correlation_matrix_float = correlation_matrix
        c_correlation_matrix_float = &mv_correlation_matrix_float[0, 0]

        # Dense correlation matrix
        _generate_matrix[float](
                coords,
                matrix_size,
                dimension,
                distance_scale,
                kernel_function,
                kernel_param,
                num_threads,
                int(verbose),
                c_correlation_matrix_float)

    elif dtype == r'float64':

        # Get pointer to the correlation matrix
        mv_correlation_matrix_double = correlation_matrix
        c_correlation_matrix_double = &mv_correlation_matrix_double[0, 0]

        # Dense correlation matrix
        _generate_matrix[double](
                coords,
                matrix_size,
                dimension,
                distance_scale,
                kernel_function,
                kernel_param,
                num_threads,
                int(verbose),
                c_correlation_matrix_double)

    elif dtype == r'float128':

        # Get pointer to the correlation matrix
        mv_correlation_matrix_long_double = correlation_matrix
        c_correlation_matrix_long_double = \
            &mv_correlation_matrix_long_double[0, 0]

        # Dense correlation matrix
        _generate_matrix[long_double](
                coords,
                matrix_size,
                dimension,
                distance_scale,
                kernel_function,
                kernel_param,
                num_threads,
                int(verbose),
                c_correlation_matrix_long_double)

    else:
        raise TypeError('"dtype" should be either "float32", "float64", or ' +
                        '"float128".')

    if verbose:
        print('Generated dense correlation matrix of size: %d.'
              % (matrix_size))

    return correlation_matrix
