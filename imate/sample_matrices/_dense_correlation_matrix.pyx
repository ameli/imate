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
from .._openmp import get_avail_num_threads

# Cython
from cython.parallel cimport parallel, prange
from ._kernels cimport get_kernel, euclidean_distance, _exponential_kernel
from libc.stdlib cimport exit, malloc, free
from libc.stdio cimport printf
from libc.math cimport NAN, sqrt, floor
from .._definitions.types cimport DataType, kernel_type
cimport cython
from .._openmp cimport omp_set_num_threads, omp_lock_t, omp_init_lock, \
    omp_set_lock, omp_unset_lock

__all__ = ['dense_correlation_matrix']

# To avoid a bug where cython does not recognize long double as a type in the
# template functions, we define long_double as an alias
ctypedef long double long_double


# ================
# get triu indices
# ================

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _get_triu_indices(
        const long int idx,
        const long int n,
        int* i_out,
        int* j_out) noexcept nogil:
    """
    Returns the i and j indices of an upper-triangular matrix given an upper
    triangular linear index.

    Parameters
    n:
        number of rows (and columns) of a square matrix.
    ind:
        the index of the upper-triangular elements, counted by iterating
        row-wise, and excluding lower-triangular elements.

    Returns
    -------

    i, j: row and column indices
    """

    cdef long int i, j

    # Initial guess based on closed-form equation
    cdef long int a = (2*n+1)**2 - 8*idx
    i = <long int> (floor(0.5 * ((2*n+1) - sqrt(<long double> a))))

    # Starting index count of row i
    cdef long int idx_row_i = i * n - (i * (i - 1) // 2)
    j = idx - idx_row_i + i

    # Adjust too large j
    while j >= n:
        i += 1
        j -= i

    # Adjust too small j
    while j < 0:
        i -= 1
        j += i

    i_out[0] = i
    j_out[0] = j


# ================
# generate element
# ================

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _generate_element(
        const double[:, ::1] coords,
        const double[:, ::1] covs,
        const long int num_points,
        const int input_dim,
        const int output_dim,
        const double scale,
        const kernel_type kernel_function,
        const double kernel_param,
        const long int num_elements,
        const long int idx,
        DataType* A) noexcept nogil:
    """
    Helper function for inside parallel loop. This function generates one
    element of the matrix.
    """
    
    cdef int i, j
    cdef int ii, jj, kk
    cdef int d = output_dim
    cdef long int n = num_points
    cdef long int m = n * d

    # Convert linear index to upper-triangular matrix index
    _get_triu_indices(idx, n, &i, &j)

    # Compute correlation
    cdef DataType rho = <DataType> kernel_function(
            euclidean_distance(
                coords[i][:],
                coords[j][:],
                scale,
                input_dim),
            kernel_param)

    # Compute linear model of co-regionalization (LMC)
    for ii in range(output_dim):
        for jj in range(output_dim):

            if (i != j) or ((i == j) and (jj >= ii)):

                A[(i*d+ii)*m + (j*d+jj)] = 0.0
                for kk in range(output_dim):
                    A[(i*d+ii)*m + (j*d+jj)] += \
                        covs[ii, i*d+kk] * covs[jj, j*d+kk]
                        
                A[(i*d+ii)*m + (j*d+jj)] = A[(i*d+ii)*m + (j*d+jj)] * rho

                # Use symmetry of the correlation matrix
                if not ((i == j) and (ii == jj)):
                    A[(j*d+jj)*m + (i*d+ii)] = A[(i*d+ii)*m + (j*d+jj)]


# ===============
# generate matrix
# ===============

@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _generate_matrix(
        const double[:, ::1] coords,
        const double[:, ::1] covs,
        const long int num_points,
        const int input_dim,
        const int output_dim,
        const double scale,
        const kernel_type kernel_function,
        const double kernel_param,
        const int num_threads,
        const int verbose,
        DataType* c_correlation_matrix) noexcept nogil:
    """
    Generates a dense correlation matrix.

    :param coords: A 2D array containing the coordinates of the spatial set of
        points in the unit hypercube. The first index of this array is the
        point ids and the second index is the dimension of the coordinates.
    :type coords: cython memoryview (double)

    :param covs: A 2D array of covariances between outputs at each point.
    Ltype covs: cython memoryview (double)

    :param matrix_size: The shape of the first index of ``coords``, which is
        also the size of the generated output matrix.
    :type matrix_size: int

    :param dimension: The shape of the second index of ``coords`` array, which
        is the dimension of the spatial points.
    :type dimension: int

    :param scale: A parameter of the correlation function that
        scales distances.
    :type scale: double

    :param nu: The parameter :math:`\\nu` of Matern correlation kernel.
    :type nu: float

    :param num_threads: Number of parallel threads in openmp.
    :type num_threads: int

    :param correlation_matrix: Output array. Correlation matrix of the size
        ``matrix_size``.
    :type correlation_matrix: cython memoryview (double)
    """

    cdef int[1] counter
    cdef int percent_update
    cdef int progress

    # Set number of parallel threads
    omp_set_num_threads(num_threads)

    # Initialize openmp lock to setup a critical section
    cdef omp_lock_t lock_counter
    omp_init_lock(&lock_counter)

    # Using max possible chunk size for parallel threads
    cdef long int idx
    cdef long int num_elements = num_points * (num_points + 1) // 2
    cdef long int chunk_size = int((<double> num_elements) / num_threads)
    if chunk_size < 1:
        chunk_size = 1

    # percent_update determines how often a progress is being printed
    if num_elements <= 50:
        percent_update = 20
    elif num_elements <= 1000:
        percent_update = 10
    elif num_elements <= 10000:
        percent_update = 5
    elif num_elements <= 50000:
        percent_update = 2
    else:
        percent_update = 1

    # Iterate over rows of correlation matrix
    counter[0] = 0
    with nogil, parallel():
        for idx in prange(num_elements, schedule='static',
                          chunksize=chunk_size):

            _generate_element[DataType](coords, covs, num_points, input_dim,
                                        output_dim, scale, kernel_function,
                                        kernel_param, num_elements, idx,
                                        c_correlation_matrix)

        # Critical section
        omp_set_lock(&lock_counter)

        # Update counter
        counter[0] = counter[0] + 1

        # Print progress on every percent_update
        if verbose and (num_elements * percent_update >= 100):
            if (counter[0] % (num_elements * percent_update // 100) == 0):
                progress = percent_update * counter[0] // \
                    (num_elements * percent_update // 100)
                printf('Generate matrix progress: %3d%%\n', progress)

        # Release lock to end the openmp critical section
        omp_unset_lock(&lock_counter)


# ========================
# dense correlation matrix
# ========================

def dense_correlation_matrix(
        coords,
        covs,
        scale=0.1,
        kernel='exponential',
        kernel_param=None,
        dtype=r'float64',
        order=r'C',
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

    :param covs: Array of covariances.
    :type covs: numpy.ndarray

    :param scale: A parameter of correlation function that scales
        distance.
    :type scale: float

    :param verbose: If ``True``, prints some information during the process.
    :type verbose: bool

    :return: Correlation matrix. If ``coords`` is ``n*m`` array, the
        correlation matrix has ``n*n`` shape.
    :rtype: numpy.ndarray

    :param nu: The parameter :math:`\\nu` of Matern correlation kernel.
    :type nu: float
    """

    # Makes kernel parameter a C-type NAN
    if kernel_param is None:
        kernel_param = NAN

    # size of data and the correlation matrix
    num_points = coords.shape[0]
    input_dim = coords.shape[1]
    output_dim = covs.shape[0]

    # Get number of CPU threads
    num_threads = get_avail_num_threads()

    # Initialize matrix
    matrix_size = num_points * output_dim
    correlation_matrix = numpy.zeros((matrix_size, matrix_size), dtype=dtype,
                                     order=order)

    # Memory view of the correlation matrix (C contiguous)
    cdef float[:, ::1] mv_c_correlation_matrix_fp32
    cdef double[:, ::1] mv_c_correlation_matrix_fp64
    cdef long double[:, ::1] mv_c_correlation_matrix_fp128

    # Memory view of the correlation matrix (F contiguous)
    cdef float[::1, :] mv_f_correlation_matrix_fp32
    cdef double[::1, :] mv_f_correlation_matrix_fp64
    cdef long double[::1, :] mv_f_correlation_matrix_fp128

    # C pointer to the correlation matrix
    cdef float* c_correlation_matrix_fp32
    cdef double* c_correlation_matrix_fp64
    cdef long double* c_correlation_matrix_fp128

    # Get the kernel function
    cdef kernel_type kernel_function = get_kernel(kernel)

    if dtype == r'float32':

        # Get pointer to the correlation matrix
        # Note: regardless of C or F order, since the matrix is symmetric, we
        # treat is the same pointer without having two separate codes.
        if order == 'C':
            mv_c_correlation_matrix_fp32 = correlation_matrix
            c_correlation_matrix_fp32 = &mv_c_correlation_matrix_fp32[0, 0]
        elif order == 'F':
            mv_f_correlation_matrix_fp32 = correlation_matrix
            c_correlation_matrix_fp32 = &mv_f_correlation_matrix_fp32[0, 0]

        # Dense correlation matrix
        _generate_matrix[float](
                coords,
                covs,
                num_points,
                input_dim,
                output_dim,
                scale,
                kernel_function,
                kernel_param,
                num_threads,
                int(verbose),
                c_correlation_matrix_fp32)

    elif dtype == r'float64':

        # Get pointer to the correlation matrix
        # Note: regardless of C or F order, since the matrix is symmetric, we
        # treat is the same pointer without having two separate codes.
        if order == 'C':
            mv_c_correlation_matrix_fp64 = correlation_matrix
            c_correlation_matrix_fp64 = &mv_c_correlation_matrix_fp64[0, 0]
        elif order == 'F':
            mv_f_correlation_matrix_fp64 = correlation_matrix
            c_correlation_matrix_fp64 = &mv_f_correlation_matrix_fp64[0, 0]

        # Dense correlation matrix
        _generate_matrix[double](
                coords,
                covs,
                num_points,
                input_dim,
                output_dim,
                scale,
                kernel_function,
                kernel_param,
                num_threads,
                int(verbose),
                c_correlation_matrix_fp64)

    elif dtype == r'float128':

        # Get pointer to the correlation matrix
        # Note: regardless of C or F order, since the matrix is symmetric, we
        # treat is the same pointer without having two separate codes.
        if order == 'C':
            mv_c_correlation_matrix_fp128 = correlation_matrix
            c_correlation_matrix_fp128 = \
                &mv_c_correlation_matrix_fp128[0, 0]
        elif order == 'F':
            mv_f_correlation_matrix_fp128 = correlation_matrix
            c_correlation_matrix_fp128 = \
                &mv_f_correlation_matrix_fp128[0, 0]

        # Dense correlation matrix
        _generate_matrix[long_double](
                coords,
                covs,
                num_points,
                input_dim,
                output_dim,
                scale,
                kernel_function,
                kernel_param,
                num_threads,
                int(verbose),
                c_correlation_matrix_fp128)

    else:
        raise TypeError('"dtype" should be either "float32", "float64", or ' +
                        '"float128".')

    if verbose:
        print('Generated dense correlation matrix of size: %d.'
              % (matrix_size))

    return correlation_matrix
