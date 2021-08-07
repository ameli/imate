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
from .._random_generator cimport py_generate_random_array
from .._definitions.types cimport DataType, IndexType, LongIndexType, FlagType
from libc.math cimport fmax, fabs, sqrt
from libc.stdio cimport printf
from libc.stdlib cimport abort


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

    .. note::

        If two vectors are identical (or the norm of their difference is very
        small), they cannot be orthogonalized against each other. In this case,
        orthogonalization is skipped against the identical vector.

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
    cdef DataType norm_v
    cdef DataType scale
    cdef DataType epsilon = 1e-15

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

        # Norm of j-th vector
        norm = cVectorOperations[DataType].euclidean_norm(
                &V[vector_size*i], vector_size)

        if norm < epsilon * sqrt(vector_size):
            printf('WARNING: norm of the given vector is too small. Cannot ')
            printf('reorthogonalize against zero vector. Skipping\n')
            break

        # Projection
        inner_prod = cVectorOperations[DataType].inner_product(
                &V[vector_size*i], v, vector_size)

        # scale for subtraction
        scale = inner_prod / (norm**2)

        # If scale is is 1, it is possible that vector v and j-th vector are
        # identical (or close).
        if fabs(scale - 1.0) <= 2.0 * epsilon:

            # Norm of the vector v
            norm_v = cVectorOperations[DataType].euclidean_norm(v, vector_size)

            # Compute distance between the j-th vector and vector v
            distance = sqrt(norm_v**2 - 2.0*inner_prod + norm**2)

            # If distance is zero, do not reorthogonalize i-th against
            # the j-th vector.
            if distance < 2.0 * epsilon * sqrt(vector_size):

                # Skip orthogonalizing against j-th vector
                continue

        # Subtraction
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

    .. note::

        If two vectors are identical (or the norm of their difference is very
        small), they cannot be orthogonalized against each other. In this case,
        one of the vectors is re-generated by new random numbers.

    .. warning::

        if ``num_vectors`` is larger than ``vector_size``, the
        orthogonalization fails since not all vectors are independent, and at
        least one vector becomes zero.

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

    cdef IndexType i = 0
    cdef IndexType j
    cdef IndexType start_j
    cdef DataType inner_prod
    cdef DataType norm
    cdef DataType norm_i
    cdef DataType distance
    cdef DataType scale
    cdef DataType epsilon = 1e-15
    cdef IndexType success = 1
    cdef IndexType max_num_trials = 20
    cdef IndexType num_trials = 0
    cdef IndexType num_threads = 1

    while i < num_vectors:

        if success == 0 and num_trials >= max_num_trials:
            printf('ERROR: Cannot orthogonalize vectors after %d trials.\n',
                   num_trials)
            abort()

        # Reset on new trial (if it was set to 0 before to start a new trial)
        success = 1

        # j iterates on previous vectors in a window of at most ``vector_size``
        if <LongIndexType>i > vector_size:
            start_j = i - vector_size
        else:
            start_j = 0

        # Reorthogonalize against previous vectors
        for j in range(start_j, i):

            # Projecting i-th vector to j-th vector
            inner_prod = cVectorOperations[DataType].inner_product(
                    &vectors[i*vector_size],
                    &vectors[j*vector_size],
                    vector_size)

            # Norm of the j-th vector
            norm = cVectorOperations[DataType].euclidean_norm(
                    &vectors[j*vector_size], vector_size)

            # Check norm
            if norm < epsilon * sqrt(vector_size):
                printf('WARNING: norm of the given vector is too small. ')
                printf('Cannot reorthogonalize against zero vector. ')
                printf('Skipping.\n')
                continue

            # Scale of subtraction
            scale = inner_prod / (norm**2)

            # If scale is is 1, it is possible that i-th and j-th vectors are
            # identical (or close). So, instead of subtracting them, regenerate
            # a new i-th vector.
            if fabs(scale - 1.0) <= 2.0 * epsilon:

                # Norm of the i-th vector
                norm_i = cVectorOperations[DataType].euclidean_norm(
                        &vectors[i*vector_size], vector_size)

                # Compute distance between i-th and j-th vector
                distance = sqrt(norm_i**2 - 2.0*inner_prod + norm**2)

                # If distance is zero, do not reorthogonalize i-th against
                # vector j-th and the subsequent vectors after j-th.
                if distance < 2.0 * epsilon * sqrt(vector_size):

                    # Regenerate new random vector for i-th vector
                    with gil:
                        py_generate_random_array(&vectors[i*vector_size],
                                                 vector_size, num_threads)

                    # Repeat the reorthogonalization for i-th vector against
                    # all previous vectors again.
                    success = 0
                    num_trials += 1
                    break

            # Subtraction
            cVectorOperations[DataType].subtract_scaled_vector(
                    &vectors[vector_size*j], vector_size, scale,
                    &vectors[vector_size*i])

            # Check norm of the i-th vector
            norm_i = cVectorOperations[DataType].euclidean_norm(
                    &vectors[i*vector_size], vector_size)

            # If the norm is too small, regenerate the i-th vector randomly
            if norm_i < epsilon * sqrt(vector_size):

                # Regenerate new random vector for i-th vector
                with gil:
                    py_generate_random_array(&vectors[i*vector_size],
                                             vector_size, num_threads)

                # Repeat the reorthogonalization for i-th vector against
                # all previous vectors again.
                success = 0
                num_trials += 1
                break

        if success == 1:
            i += 1

            # Reset if num_trials was incremented before.
            num_trials = 0
