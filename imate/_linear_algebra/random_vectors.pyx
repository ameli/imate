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
from .._definitions.types cimport DataType, IndexType, LongIndexType
from .._random_generator cimport py_generate_random_array
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
        const IndexType num_threads):
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

    # Generate random array
    cdef LongIndexType vectors_size = vector_size * num_vectors
    py_generate_random_array(vectors, vectors_size, num_threads)

    # Orthogonalize (optional). This section is not parallel (must be serial)
    if orthogonalize:
        orthogonalize_vectors(vectors, vector_size, num_vectors)

    # Set the number of threads
    openmp.omp_set_num_threads(num_threads)

    # Using max possible chunk size for parallel schedules
    cdef IndexType chunk_size = int((<DataType> num_vectors) / num_threads)
    if chunk_size < 1:
        chunk_size = 1

    # Normalize (necessary)
    cdef IndexType j
    with nogil, parallel():
        for j in prange(num_vectors, schedule='static', chunksize=chunk_size):
            cVectorOperations[DataType].normalize_vector_in_place(
                    &vectors[j*vector_size], vector_size)
