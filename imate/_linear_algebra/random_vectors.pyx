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

from cython.parallel cimport parallel, prange
from libc.time cimport time, time_t
from libc.stdlib cimport rand, srand, RAND_MAX
from libc.math cimport copysign
from .._definitions.types cimport DataType, IndexType, LongIndexType
from .orthogonalization cimport orthogonalize_vectors
from .._c_basic_algebra cimport cVectorOperations
cimport openmp


# ==============================
# generate random column vectors
# ==============================

cdef void generate_random_column_vectors(
        DataType* vectors,
        const LongIndexType vector_size,
        const IndexType num_vectors,
        const IndexType orthogonalize,
        const IndexType num_threads) nogil:
    """
    Generates a set of random column vectors using Rademacher distribution. The
    column vectors are normalized to unit norm. If desired, the vectors can
    also be orthogonalized by setting ``orthogonalize`` flag to a non-zero
    value. The computation is parallelized using OpenMP over the number of
    column vectors.

    :param vectors: 2D array of vectors of size ``(vector_size, num_vectors)``.
        This array will be written in-place and serves as the output. Note this
        is Fortran ordering, meaning that the first index is contiguous. To
        refer to the j-th column vector, use ``&vectors[0][j]``.
    :type vectors: pointer (DataType)

    :param vector_size: Number of rows of vectors array.
    :type vector_size: LongIndexType

    :param num_vectors: Number of columns of vectors array.
    :type num_vectors: IndexType

    :param orthogonalize: A flag, when set to a non-zero value, the vectors are
        orthogonalized using modified Gram-Schmidt process. Otherwise, the
        generated vectors remain enact.
    :param orthogonalize: IndexType

    :param num_threads: Number of OpenMP parallel threads
    :type num_threads: IndexType
    """

    # Set the seed of rand function
    cdef time_t t
    srand((<IndexType> time(&t)))

    # Set the number of threads
    openmp.omp_set_num_threads(num_threads)

    # Using max possible chunk size for parallel schedules
    cdef IndexType chunk_size = int((<DataType> num_vectors) / num_threads)
    if chunk_size < 1:
        chunk_size = 1

    # Shared-memory parallelism over individual column vectors
    cdef LongIndexType i
    cdef IndexType j
    with nogil, parallel():
        for j in prange(num_vectors, schedule='static', chunksize=chunk_size):

            # Inner iteration over the contiguous elements of vector array
            for i in range(vector_size):

                # Produce Rademacher distribution (+1 and -1 with uniform dist)
                vectors[j*vector_size + i] = copysign(
                        1.0,
                        (2.0 * rand() / (<DataType>RAND_MAX)) - 1.0)

    # Orthogonalize (optional). This section is not parallel (must be serial)
    if orthogonalize:
        orthogonalize_vectors(vectors, vector_size, num_vectors)

    # Normalize (necessary)
    with nogil, parallel():
        for j in prange(num_vectors, schedule='static', chunksize=chunk_size):
            cVectorOperations[DataType].normalize_vector_in_place(
                    &vectors[j*vector_size], vector_size)


# ===========================
# generate random row vectors
# ===========================

cdef void generate_random_row_vectors(
        DataType** vectors,
        const LongIndexType vector_size,
        const IndexType num_vectors,
        const IndexType num_threads) nogil:
    """
    Generates a set of random row vectors using Rademacher distribution. The
    row vectors are normalized to unit norm. The computation is parallelized
    using OpenMP over the number of vectors.

    :param vectors: 1D array representing 2D array of the shape
        ``(num_vectors, vector_size)``, which represents a set of row vectors.
        This array will be written in-place and serves as the output.
        To refer to the i-th vector, use ``&vectors[i][0]``. Note that this
        2D array has C (row major) ordering, meaning that the last index is
        contiguous.
    :type vectors: pointer (DataType)

    :param vector_size: Number of rows of vectors array.
    :type vector_size: LongIndexType

    :param num_vectors: Number of columns of vectors array.
    :type num_vectors: IndexType

    :param num_threads: Number of OpenMP parallel threads
    :type num_threads: IndexType
    """

    # Set the seed of rand function
    cdef time_t t
    srand((<IndexType> time(&t)))

    # Set the number of threads
    openmp.omp_set_num_threads(num_threads)

    # Using max possible chunk size for parallel schedules
    cdef IndexType chunk_size = int((<DataType> num_vectors) / num_threads)
    if chunk_size < 1:
        chunk_size = 1

    # Shared-memory parallelism over individual row vectors
    cdef IndexType i
    cdef LongIndexType j
    with nogil, parallel():
        for i in prange(num_vectors, schedule='static', chunksize=chunk_size):

            # Inner iteration over the contiguous elements of vector array
            for j in range(vector_size):

                # Produce Rademacher distribution (+1 and -1 with uniform dist)
                vectors[i][j] = copysign(
                        1.0,
                        (2.0 * rand() / (<DataType>RAND_MAX)) - 1.0)

            # Normalize
            cVectorOperations[DataType].normalize_vector_in_place(
                    &vectors[i][0], vector_size)
