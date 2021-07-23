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

from .._c_basic_algebra cimport cVectorOperations
from .._definitions.types cimport DataType, IndexType, LongIndexType, FlagType
from libc.math cimport fmax


# ====================
# gram schmidt process
# ====================

cdef void gram_schmidt_process(
        const DataType* V,
        const LongIndexType vector_size,
        const IndexType num_vectors,
        const FlagType ortho_depth,
        DataType* v) nogil:
    """
    Modified Gram-Schmidt orthogonalization process to orthogonalize the vector
    ``v`` against a subset of the column vectors in the array ``V``.

    ``V`` is 1D array of the length ``vector_size * num_vectors`` to represent
    a 2D array of a set of ``num_vectors`` column vectors, each of the length
    ``vector_size``. The length of ``v`` is also ``vector_size``.

    ``v`` is orthogonalized against the last ``ortho_depth`` columns of ``V``.

        * If ``ortho_depth`` is zero, no orthogonalization is performed.
        * If ``ortho-depth`` is negative (usually set to ``-1``), then ``v`` is
          orthogonalized against all column vectors of ``V``.
        * If `ortho_depth`` is larger than ``num_vectors``, then ``v`` is
          orthogonalized against all column vectors of ``V``.
        * If ``ortho-depth`` is smaller than ``num_vectors``, then ``v`` is
          orthogonalized against the last ``orth-depth`` column vectors of
          ``V``.

    The result of the newer ``v`` is written in-place in ``v``.

    :param V: 1D coalesced  array of vectors representing a 2D array.
        The length of this 1D array is vector_size*num_vectors, which indicates
        a 2D array with the shape ``(vector_size, num_vectors)``.
    :type V: C pointer (DataType)

    :param vector_size: The length of each vector. If we assume ``V`` indicates
        a 2D vector, this the number of rows of ``V``.
    :type vector_size: int

    :param num_vector: The number of column vectors. If we assume ``V``
        indicates a 2D vector, this the number of columns of ``V``.
    :type num_vectors: int

    :param ortho_depth: The number of vectors to be orthogonalized starting
        from the last vector. ``0`` indicates no orthogonalization will be
        performed and the function just returns. A negative value means all
        vectors will be orthogonalized. A poisitive value will orthogonalize
        the given number of vectors. This value cannot be larger than the
        number of vectors.
    :type ortho_depth: FlagType

    :param v: The vector that will be orthogonalized against the columns of
        `V``. The length of ``v`` is ``vector_size``. This vector is modified
        in-place.
    :param v: C pointer (DataType)

    :param vector_size: The length of each vector. This the number of rows of
        ``basis`` as a 2D array.
    :type vector_size: int

    :param num_vector: The number of column vectors. This the number of columns
        of ``basis`` as a 2D array.
    :type num_vectors: int
    """

    cdef IndexType i
    cdef LongIndexType j
    cdef IndexType num_steps
    cdef DataType inner_prod
    cdef DataType norm
    cdef DataType scale

    # Determine how many previous vectors to orthogonalize against
    if ortho_depth == 0:
        # No orthogonalization is performed
        return

    elif (ortho_depth < 0) or (ortho_depth > <FlagType> num_vectors):
        # Orthogonalize against all vectors
        num_steps = num_vectors

    else:
        # Orthogonalize against only the last ortho_depth vectors
        num_steps = ortho_depth

    # Vectors can be orthogonalized at most to the full basis of the vector
    # space. Thus, num_steps cannot be larger than the dimension of vector
    # space, which is vector_size.
    if num_steps > <IndexType>(vector_size):
        num_steps = vector_size

    # Iterate over vectors
    for i in range(num_vectors-1, num_vectors-num_steps-1, -1):

        # Projection
        inner_prod = cVectorOperations[DataType].inner_product(
                &V[vector_size*i], v, vector_size)

        norm = cVectorOperations[DataType].euclidean_norm(
                &V[vector_size*i], vector_size)

        # Subtraction
        scale = inner_prod / (norm**2)
        cVectorOperations[DataType].subtract_scaled_vector(
                &V[vector_size*i], vector_size, scale, v)


# =====================
# orthogonalize vectors
# =====================

cdef void orthogonalize_vectors(
        DataType* vectors,
        const LongIndexType vector_size,
        const IndexType num_vectors) nogil:
    """
    Orthogonalizes set of vectors mutually using modified Gram-Schmidt process.

    .. note::

        Let ``m`` be the number of vectors (``num_vectors``), and let ``n`` be
        the size of each vector (``vector_size``). In general, ``n`` is much
        larger (large matrix size), and ``m`` is small, in order of a couple of
        hundred. But for small matrices (where ``n`` could be smaller then
        ``m``), then each vector can be orthogonalized at most to ``n`` other
        vectors. This is because the dimension of the vector space is ``n``.
        Thus, if there are extra vectors, each vector is orthogonalized to
        window of the previous ``n`` vector.

    :param vectors: 2D array of size ``vector_size * num_vectors``. This array
        will be modified in-place and will be output of this function. Note
        this is Fortran ordering, meaning that the first index is contiguous.
        Hence, to call the i-th vector, use ``&vectors[0][i]``. Here, iteration
        over the first index is continuous.
    :type vectors: pointer (DataType)

    :param vector_size: Number of rows of vectors array.
    :type vector_size: IndexType

    :param num_vectors: Number of columns of vectors array.
    :type num_vectors: unsigned long
    """

    # Do nothing if there is only one vector
    if num_vectors < 2:
        return

    cdef IndexType i, j
    cdef IndexType start_j
    cdef DataType inner_prod
    cdef DataType norm
    cdef DataType scale

    for i in range(num_vectors):

        # j iterates on previous vectors in a window of at most ``vector_size``
        if <LongIndexType>i > vector_size:
            start_j = i - vector_size
        else:
            start_j = 0

        for j in range(start_j, i):

            # Projecting i-th vector to j-th vector
            inner_prod = cVectorOperations[DataType].inner_product(
                    &vectors[i*vector_size],
                    &vectors[j*vector_size],
                    vector_size)

            # Norm of the j-th vector
            norm = cVectorOperations[DataType].euclidean_norm(
                    &vectors[j*vector_size], vector_size)

            # Subtraction
            scale = inner_prod / (norm**2)
            cVectorOperations[DataType].subtract_scaled_vector(
                    &vectors[vector_size*j], vector_size, scale,
                    &vectors[vector_size*i])
