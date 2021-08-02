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
import scipy
from scipy import sparse
import multiprocessing

# Cython
from cython.parallel cimport parallel, prange
from ._kernels cimport get_kernel, euclidean_distance
from libc.stdio cimport printf
from libc.stdlib cimport exit, malloc, free
from libc.math cimport NAN
from .._definitions.types cimport DataType, kernel_type
cimport cython
cimport openmp

__all__ = ['sparse_correlation_matrix']

# To avoid a bug where cython does not recognize long doubler as a type in the
# template functions, we define long_double as an alias
ctypedef long double long_double


# ===============
# generate matrix
# ===============

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int _generate_matrix(
        const double[:, ::1] coords,
        const int matrix_size,
        const int dimension,
        const double distance_scale,
        const kernel_type kernel_function,
        const double kernel_param,
        const double kernel_threshold,
        const int num_threads,
        const int verbose,
        const long max_nnz,
        long[:] nnz,
        long[:] matrix_row_indices,
        long[:] matrix_column_indices,
        DataType* c_matrix_data) nogil:
    """
    Generates a sparse correlation matrix.

    In this function, we pre-allocated array of sparse matrix with presumed nnz
    equal to ``max_nnz``. If the number of required nnz is more than that, this
    function should be stopped and a newer array with larger memory should be
    pre-allocated. To stop openmp loop, we cannot use ``break``. Instead we use
    the ``success`` variable. A zero success variable signals other threads to
    perform idle loops till end. This way, the openmp is terminated gracefully.

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

    :param kernel_threshold: The parameter tapers the correlation kernel. The
        kernel values below kernel threshold are assumed to be zero, which
        sparsifies the matrix.
    :type kernel_threshold: double

    :param num_threads: Number of parallel threads in openmp.
    :type num_threads: int

    :param max_nnz: The size of pre-allocated sparse matrix. The generated
        sparse matrix many have less or more nnz than this value. If the matrix
        requires more nnz, max_nnz should be increased accordingly and this
        function should be recalled.
    :type max_nnz: long

    :param nnz: An output variable, showing the actual nnz of the generated
        sparse matrix.
    :type nnz: long

    :param matrix_row_indices: The row indices of sparse matrix in COO format.
        The size of this array is equal to max_nnz. However, in practice, the
        matrix may have smaller nnz.
    :type matrix_row_indices: cython memoryview (long)

    :param matrix_column_indices: The column indices of sparse matrix in COO
        format. The size of this array is equal to max_nnz. However, in
        practice, the matrix may have smaller nnz.
    :type matrix_column_indices: cython memoryview (long)

    :param c_matrix_data: The non-zero data of sparse matrix in COO format. The
        size of this array is max_nnz.
    :type c_matrix_data: cython memoryview (double)

    :return: success of the process. If the required nnz to generate the sparse
        matrix needs to be larger than the preassigned max_nnz, this function
        is terminated and ``0`` is returned. However, if the required nnz is
        less than or equal to max_nnz and the sparse matrix is generated
        successfully, ``1`` (success) is returned.
    :rtype: int
    """

    cdef long i, j
    cdef int dim
    cdef double *thread_data = <double *> malloc(num_threads * sizeof(double))
    cdef int[1] success
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
    cdef openmp.omp_lock_t lock
    openmp.omp_init_lock(&lock)

    # Initialize an openmp lock for the counter and printing the progress
    cdef openmp.omp_lock_t lock_counter
    openmp.omp_init_lock(&lock_counter)

    # Using max possible chunk size for parallel threads
    cdef int chunk_size = int((<double> matrix_size) / num_threads)
    if chunk_size < 1:
        chunk_size = 1

    # Iterate over rows of matrix
    nnz[0] = 0
    success[0] = 1
    counter[0] = 0
    with nogil, parallel():
        for i in prange(matrix_size, schedule='dynamic', chunksize=chunk_size):

            # If one of threads was not successful, do empty loops till end
            # we cannot break openmp loop, so here we just continue
            if not success[0]:
                continue

            for j in range(i, matrix_size):

                # Compute an element of the matrix
                thread_data[openmp.omp_get_thread_num()] = kernel_function(
                        euclidean_distance(
                            coords[i][:],
                            coords[j][:],
                            distance_scale,
                            dimension),
                        kernel_param)

                # Check with kernel threshold to taper out or store
                if thread_data[openmp.omp_get_thread_num()] > kernel_threshold:

                    # Halt operation if preallocated sparse array is not enough
                    if nnz[0] >= max_nnz:

                        # Critical section
                        openmp.omp_set_lock(&lock)

                        # Avoid duplicate if success is falsified already
                        if success[0]:
                            printf('Pre-allocated sparse array reached max ')
                            printf('nnz: %ld. Terminate operation.\n', max_nnz)
                            success[0] = 0

                        # Release lock to end the openmp critical section
                        openmp.omp_unset_lock(&lock)

                        # The inner loop is not an openmp loop, so we can break
                        break

                    # Add data to the arrays in an openmp critical section
                    openmp.omp_set_lock(&lock)

                    nnz[0] += 1
                    matrix_row_indices[nnz[0]-1] = i
                    matrix_column_indices[nnz[0]-1] = j
                    c_matrix_data[nnz[0]-1] = \
                        thread_data[openmp.omp_get_thread_num()]

                    # Use symmetry of the matrix
                    if i != j:
                        nnz[0] += 1
                        matrix_row_indices[nnz[0]-1] = j
                        matrix_column_indices[nnz[0]-1] = i
                        c_matrix_data[nnz[0]-1] = \
                            thread_data[openmp.omp_get_thread_num()]

                    # Release lock to end the openmp critical section
                    openmp.omp_unset_lock(&lock)

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

    free(thread_data)

    return success[0]


# ===========
# ball radius
# ===========

def _ball_radius(volume, dimension):
    """
    Computes the radius of n-ball at dimension n, given its volume.

    :param volume: Volume of n-ball
    :type volume: double

    :param dimension: Dimension of embedding space
    :type dimension: int

    :return: radius of n-ball
    :rtype: double
    """

    # Compute Gamma(dimension/2 + 1)
    if dimension % 2 == 0:
        k = 0.5 * dimension
        gamma = 1.0
        while k > 0.0:
            gamma *= k
            k -= 1.0
    else:
        k = numpy.ceil(0.5 * dimension)
        gamma = numpy.sqrt(numpy.pi)
        while k > 0.0:
            gamma *= k - 0.5
            k -= 1.0

    # radius from volume
    radius = (gamma * volume)**(1.0 / dimension) / numpy.sqrt(numpy.pi)

    return radius


# =========================
# estimate kernel threshold
# =========================

def _estimate_kernel_threshold(
        matrix_size,
        dimension,
        density,
        distance_scale,
        kernel,
        kernel_param):
    """
    Estimates the kernel's tapering threshold to sparsify a dense matrix into a
    sparse matrix with the requested density.

    Here is how density :math:`\\rho` is related to the kernel_threshold
    :math:`\\tau`:

    .. math::

        a = \\rho n = \\mathrm{Vol}_{d}(r/l),
        \\tau = k(r),

    where:

        * :math:`n` is the number of points in the unit hypercube, also it is
          the matrix size.
        * :math:`d` is the dimension of space.
        * :math:`\\mathrm{Vol}_{d}(r/l)` is the volume of d-ball of radius
          :math:`r/l`.
        * :math:`l = 1/(\\sqrt[d]{n} - 1)` is the grid size along each axis,
          assuming the points are places on an equi-distanced structured grid.
        * :math:`k` is the Matern correlation function.
        * :math:`a` is the adjacency of a point, which is the number of
          the neighbor points that are correlated to a point.
        * :math:`\\rho` is the sparse matrix density (input to this function).
        * :math:`\\tau` is the kernel threshold (output of this function).

    The adjacency :math:`a` is the number of points on an integer lattice
    and inside a d-ball. This quantity can be approximated by the volume of a
    d-ball, see for instance
    `Gauss circle problem<https://en.wikipedia.org/wiki/Gauss_circle_problem>`_
     in 2D.

    A non-zero kernel threshold is used to sparsify a matrix by tapering its
    correlation function. However, if the kernel threshold is too large, some
    elements of the correlation matrix will not be correlated to any other
    neighbor point. This leads to a correlation matrix with some rows that have
    only one non-zero element equal to one on the diagonal and zero elsewhere.
    Essentially, if all points loose their correlation to a neighbor, the
    matrix becomes identity.

    This function checks if a set of parameters to form a sparse matrix could
    lead to this issue. The correlation matrix in this module is formed by the
    mutual correlation between spatial set of points in the unit hypercube. We
    assume each point is surrounded by a sphere of the radius of the kernel
    threshold. If this radius is large enough, then the sphere of all points
    intersect. If the criteria is not met, this function raises ``ValueError``.

    :param matrix_size: The size of the square matrix. This is also the number
        of points used to construct the correlation matrix.
    :type matrix_size: int

    :param dimension: The dimension of the space of points used to construct
        the correlation matrix.
    :type dimension: int

    :param sparse_density: The desired density of the sparse matrix. Note that
        the actual density of the generated matrix will not be exactly equal to
        this value. If the matrix size is large, this value is close to the
        actual matrix density.
    :type sparse_density: int

    :param distance_scale: A parameter of correlation function that scales
        distance.
    :type distance_scale: float

    :param nu: The parameter :math:`\\nu` of Matern correlation kernel.
    :type nu: float

    :return: Kernel threshold level
    :rtype: double
    """

    # Number of neighbor points to be correlated in a neighborhood of a point
    adjacency_volume = density * matrix_size

    # If Adjacency is less that one, the correlation matrix becomes identity
    # since no point will be adjacent to other in the correlation matrix.
    if adjacency_volume < 1.0:
        raise ValueError(
                'Adjacency: %0.2f. Correlation matrix will become identity '
                % (adjacency_volume) +
                'since kernel radius is less than grid size. To increase ' +
                'adjacency, consider increasing density or distance_scale.')

    # Approximate radius of n-sphere containing the above number of adjacent
    # points, assuming adjacent points are distanced on integer lattice.
    adjacency_radius = _ball_radius(adjacency_volume, dimension)

    # Number of points along each axis of the grid
    grid_axis_num_points = matrix_size**(1.0 / dimension)

    # Size of grid elements
    grid_size = 1.0 / (grid_axis_num_points - 1.0)

    # Scale the integer lattice of adjacency radius by the grid size.
    # This is the tapering radius of the kernel
    kernel_radius = grid_size * adjacency_radius

    # Get the kernel functon
    cdef kernel_type kernel_function = get_kernel(kernel)

    # Threshold of kernel to perform tapering
    kernel_threshold = kernel_function(kernel_radius/distance_scale,
                                       kernel_param)

    return kernel_threshold


# ================
# estimate max nnz
# ================

def _estimate_max_nnz(
        matrix_size,
        dimension,
        density):
    """
    Estimates the maximum number of nnz needed to store the indices and data of
    the generated sparse matrix. Before the generation of the sparse matrix,
    its nnz (number of non-zero elements) are not known. Thus, this function
    only guesses this value based on its density.

    :param matrix_size: The size of the square matrix. This is also the number
        of points used to construct the correlation matrix.
    :type matrix_size: int

    :param dimension: The dimension of the space of points used to construct
        the correlation matrix.
    :type dimension: int

    :param sparse_density: The desired density of the sparse matrix. Note that
        the actual density of the generated matrix will not be exactly equal to
        this value. If the matrix size is large, this value is close to the
        actual matrix density.
    :type sparse_density: int

    :return: maximum non-zero elements of sparse array
    :rtype: double
    """

    estimated_nnz = int(numpy.ceil(density * (matrix_size**2)))

    # Multiply the estimated nnz by unit hypercube over unit ball volume ratio
    safty_coeff = 1.0 / _ball_radius(1.0, dimension)
    max_nnz = int(numpy.ceil(safty_coeff * estimated_nnz))

    return max_nnz


# =========================
# sparse correlation matrix
# =========================

def sparse_correlation_matrix(
        coords,
        distance_scale=0.1,
        kernel='exponential',
        kernel_param=None,
        density=0.001,
        dtype=r'float64',
        verbose=False):
    """
    Generates either a sparse correlation matrix in CSR format.

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

    :param nu: The parameter :math:`\\nu` of Matern correlation kernel.
    :type nu: float

    :param density: The desired density of the non-zero elements of the sparse
        matrix. Note that the actual density of the generated matrix may not be
        exactly equal to this value. If the matrix size is large, this value is
        close to the actual matrix density.
    :type sparse_density: int

    :param verbose: If ``True``, prints some information during the process.
    :type verbose: bool

    :return: Correlation matrix. If ``coords`` is ``n*m`` array, the
        correlation ``K`` is ``n*n`` matrix.
    :rtype: numpy.ndarray or scipy.sparse array

    .. warning::
        ``kernel_threshold`` should be large enough so that the correlation
        matrix is not shrunk to identity. If such case happens, this function
        raises a *ValueError* exception.

        In addition, ``kernel_threshold`` should be small enough to not
        eradicate its positive-definiteness. This is not checked by this
        function and the user should be aware of it.
    """

    # Makes kernel paramerter a C-type NAN
    if kernel_param is None:
        kernel_param = NAN

    # size of data and the correlation matrix
    matrix_size = coords.shape[0]
    dimension = coords.shape[1]

    # Get number of CPU threads
    num_threads = multiprocessing.cpu_count()

    # Get the kernel functon
    cdef kernel_type kernel_function = get_kernel(kernel)

    # kernel threshold
    kernel_threshold = _estimate_kernel_threshold(
            matrix_size,
            dimension,
            density,
            distance_scale,
            kernel,
            kernel_param)

    # maximum nnz
    max_nnz = _estimate_max_nnz(
            matrix_size,
            dimension,
            density)

    # Memory view of the correlation matrix
    cdef float[:] mv_matrix_data_float
    cdef double[:] mv_matrix_data_double
    cdef long double[:] mv_matrix_data_long_double

    # C pointer to the correlation matrix
    cdef float* c_matrix_data_float
    cdef double* c_matrix_data_double
    cdef long double* c_matrix_data_long_double

    # Try with the estimated nnz. If not enough, we will double and retry
    success = 0

    while not bool(success):

        # Allocate sparse arrays
        matrix_row_indices = numpy.zeros((max_nnz,), dtype=int)
        matrix_column_indices = numpy.zeros((max_nnz,), dtype=int)
        matrix_data = numpy.zeros((max_nnz,), dtype=dtype)
        nnz = numpy.zeros((1,), dtype=int)

        if dtype == r'float32':

            # Get pointer to the correlation matrix
            mv_matrix_data_float = matrix_data
            c_matrix_data_float = &mv_matrix_data_float[0]

            # Generate matrix assuming the estimated nnz is enough
            success = _generate_matrix[float](
                    coords,
                    matrix_size,
                    dimension,
                    distance_scale,
                    kernel_function,
                    kernel_param,
                    kernel_threshold,
                    num_threads,
                    int(verbose),
                    max_nnz,
                    nnz,
                    matrix_row_indices,
                    matrix_column_indices,
                    c_matrix_data_float)

        elif dtype == r'float64':

            # Get pointer to the correlation matrix
            mv_matrix_data_double = matrix_data
            c_matrix_data_double = &mv_matrix_data_double[0]

            # Generate matrix assuming the estimated nnz is enough
            success = _generate_matrix[double](
                    coords,
                    matrix_size,
                    dimension,
                    distance_scale,
                    kernel_function,
                    kernel_param,
                    kernel_threshold,
                    num_threads,
                    int(verbose),
                    max_nnz,
                    nnz,
                    matrix_row_indices,
                    matrix_column_indices,
                    c_matrix_data_double)

        elif dtype == r'float128':

            # Get pointer to the correlation matrix
            mv_matrix_data_long_double = matrix_data
            c_matrix_data_long_double = &mv_matrix_data_long_double[0]

            # Generate matrix assuming the estimated nnz is enough
            success = _generate_matrix[long_double](
                    coords,
                    matrix_size,
                    dimension,
                    distance_scale,
                    kernel_function,
                    kernel_param,
                    kernel_threshold,
                    num_threads,
                    int(verbose),
                    max_nnz,
                    nnz,
                    matrix_row_indices,
                    matrix_column_indices,
                    c_matrix_data_long_double)

        else:
            raise TypeError('"dtype" should be either "float32", "float64" ' +
                            ', or "float128".')

        # Double the number of pre-allocated nnz and try again
        if not bool(success):
            max_nnz = 2 * max_nnz

            if verbose:
                print('Retry generation of sparse matrix with max_nnz: %d'
                      % (max_nnz))

    # Construct scipy.sparse.coo_matrix, then convert it to CSR matrix.
    correlation_matrix = scipy.sparse.coo_matrix(
            (matrix_data[:nnz[0]],
             (matrix_row_indices[:nnz[0]],
              matrix_column_indices[:nnz[0]])),
            shape=(matrix_size, matrix_size)).tocsr()

    # Actual sparse density
    if verbose:
        actual_density = correlation_matrix.nnz / \
                numpy.prod(correlation_matrix.shape)
        print('Generated sparse correlation matrix using ' +
              'kernel threshold: %0.4f and ' % (kernel_threshold) +
              'sparse density: %0.2e.' % (actual_density))

    return correlation_matrix
