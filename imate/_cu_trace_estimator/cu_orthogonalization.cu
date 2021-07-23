/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */


// =======
// Imports
// =======

#include "./cu_orthogonalization.h"
#include "../_cu_basic_algebra/cu_vector_operations.h"  // cuVectorOperations


// ====================
// gram schmidt process
// ====================

/// \brief         Modified Gram-Schmidt orthogonalization process to
///                orthogonalize the vector \c v against a subset of the column
///                vectors in the array \c V.
///
/// \details       \c V is 1D array of the length \c vector_size*num_vectors to
///                represent a 2D array of a set of \c num_vectors column
///                vectors, each of the length \c vector_size. The length of
///                \c v is also \c vector_size.
///
///                \c v is orthogonalized against the last \c num_ortho
///                columns of \c V starting from the column vector of the index
///                \c last_vector. If the backward indexing from \c last_vector
///                becomes a negative index, the index wraps around from the
///                last column vector index, i.e., \c num_vectors-1 .
///
///                * If \c num_ortho is zero, or if \c num_vectors is zero, no
///                  orthogonalization is performed.
///                * If \c num_ortho is negative (usually set to \c -1), then
///                  \c v is orthogonalized against all column vectors of \c V.
///                * If \c num_ortho is larger than \c num_vectors, then \c v
///                  is orthogonalized against all column vectors of \c V.
///                * If \c num_ortho is smaller than \c num_vectors, then
///                  \c v is orthogonalized against the last \c num_ortho
///                  column vectors of \c V, starting from the column vector
///                  with the index \c last_vector toward its previous vectors.
///                  If the iteration runs into negativen column indices, the
///                  column indexing wraps around from the end of the columns
///                  from \c num_vectors-1.
///
///                The result of the newer \c v is written in-place in \c v.
///
/// \note          It is assumed that the caller function fills the column
///                vectors of \c V periodically in a *wrapped around* order
///                from column index \c 0,1,... to \c num_vectors-1, and newer
///                vectors are replaced on the wrapped index starting from
///                index \c 0,1,... again. Thus, \c V only stores the last
///                \c num_vectors column vectors. The index of the last filled
///                vector is indicated by \c last_vector.
///
/// \warning       The vector \c v can be indeed one of the columns of \c V
///                itself. However, in this case, vector \c v must *NOT* be
///                orthogonalized against itself, rather, it should only be
///                orthogonalized to the other vectors in \c V. For instance,
///                if \c num_vectors=10, and \c v is the 3rd vector of \c V,
///                and if \c num_ortho is \c 6, then we may set
///                \c last_vector=2. Then \c v is orthogonalized againts the
///                six columns \c 2,1,0,9,8,7, where the last three of them are
///                wrapped around the end of the columns.
///
/// \sa            cu_golub_kahn_bidiagonalizaton,
///                cu_lanczos_bidiagonalization
///
/// \param[in]     V
///                1D coalesced array of vectors representing a 2D array. The
///                length of this 1D array is \c vector_size*num_vectors, which
///                indicates a 2D array with the shape
///                \c (vector_size,num_vectors).
/// \param[in]     vector_size
///                The length of each vector. If we assume \c V indicates a 2D
///                vector, this is the number of rows of \c V.
/// \param[in]     num_vector
///                The number of column vectors. If we assume \c V indicates a
///                2D vector, this the number of columns of \c V.
/// \param[in]     last_vector
///                The column vectors of the array \c V are rewritten by the
///                caller function in wrapped-around order. That is, once all
///                the columns (from the zeroth to the \c num_vector-1 vector)
///                are filled, the next vector is rewritten in the place of
///                the zeroth vector, and the indices of newer vectors wrap
///                around the columns of \c V. Thus, \c V only retains the last
///                \c num_vectors vectors. The column index of the last written
///                vector is given by \c last_vector. This index is a number
///                between \c 0 and \c num_vectors-1. The index of the last
///                i-th vector is winding back from the last vector by
///                $last_vector-i+1 \mod num_vectors$.
/// \param[in]     num_ortho
///                The number of vectors to be orthogonalized starting from the
///                last vector. \c 0 indicates no orthogonalization will be
///                performed and the function just returns. A negative value
///                means all vectors will be orthogonalized. A poisitive value
///                will orthogonalize the given number of vectors. This value
///                cannot be larger than the number of vectors.
/// \param[in,out] v
///                The vector that will be orthogonalized against the columns
///                of \c V. The length of \c v is \c vector_size. This vector
///                is modified in-place.

template <typename DataType>
void cuOrthogonalization<DataType>::gram_schmidt_process(
        cublasHandle_t cublas_handle,
        const DataType* V,
        const LongIndexType vector_size,
        const IndexType num_vectors,
        const IndexType last_vector,
        const FlagType num_ortho,
        DataType* v)
{
    // Determine how many previous vectors to orthogonalize against
    IndexType num_steps;
    if ((num_ortho == 0) || (num_vectors < 2))
    {
        // No orthogonalization is performed
        return;
    }
    else if ((num_ortho < 0) ||
             (num_ortho > static_cast<FlagType>(num_vectors)))
    {
        // Orthogonalize against all vectors
        num_steps = num_vectors;
    }
    else
    {
        // Orthogonalize against only the last num_ortho vectors
        num_steps = num_ortho;
    }

    // Vectors can be orthogonalized at most to the full basis of the vector
    // space. Thus, num_steps cannot be larger than the dimension of vector
    // space, which is vector_size.
    if (num_steps > static_cast<IndexType>(vector_size))
    {
        num_steps = vector_size;
    }

    IndexType i;
    DataType inner_prod;
    DataType norm;

    // Iterate over vectors
    for (IndexType step=0; step < num_steps; ++step)
    {
        // i is the index of a column vector in V to orthogonalize v against it
        if ((last_vector % num_vectors) >= step)
        {
            i = (last_vector % num_vectors) - step;
        }
        else
        {
            // Wrap around negative indices from the end of column index
            i = (last_vector % num_vectors) - step + num_vectors;
        }

        // Projection
        inner_prod = cuVectorOperations<DataType>::inner_product(
                cublas_handle, &V[vector_size*i], v, vector_size);
        norm = cuVectorOperations<DataType>::euclidean_norm(
                cublas_handle, &V[vector_size*i], vector_size);

        // Subtraction
        DataType scale = inner_prod / (norm * norm);
        cuVectorOperations<DataType>::subtract_scaled_vector(
                cublas_handle, &V[vector_size*i], vector_size, scale, v);
    }
}


// =====================
// orthogonalize vectors
// =====================

/// \brief         Orthogonalizes set of vectors mutually using modified
///                Gram-Schmidt process.
///
/// \note          Let \c m be the number of vectors (\c num_vectors), and
///                let \c n be the size of each vector (\c vector_size). In
///                general, \c n is much larger (large matrix size), and \c m
///                is small, in order of a couple of hundred. But for small
///                matrices (where \c n could be smaller then \c m), then each
///                vector can be orthogonalized at most to \c n other vectors.
///                This is because the dimension of the vector space is \c n.
///                Thus, if there are extra vectors, each vector is
///                orthogonalized to window of the previous \c n vector.
///
/// \param[in,out] vectors
///                2D array of size \c vector_size*num_vectors. This array will
///                be modified in-place and will be output of this function.
///                Note that this is Fortran ordering, meaning that the first
///                index is contiguous. Hence, to call the j-th element of the
///                i-th vector, use \c &vectors[i*vector_size + j].
/// \param[in]     num_vectors
///                Number of columns of vectors array.
/// \param[in]     vector_size
///                Number of rows of vectors array.

template <typename DataType>
void cuOrthogonalization<DataType>::orthogonalize_vectors(
        cublasHandle_t cublas_handle,
        DataType* vectors,
        const LongIndexType vector_size,
        const IndexType num_vectors)
{
    // Do nothing if there is only one vector
    if (num_vectors < 2)
    {
        return;
    }

    IndexType i;
    IndexType j;
    IndexType start = 0;
    DataType inner_prod;
    DataType norm;

    for (i=0; i < num_vectors; ++i)
    {
        // j iterates on previous vectors in a window of at most vector_size
        if (static_cast<LongIndexType>(i) > vector_size)
        {
            // When vector_size is smaller than i, it is fine to cast to signed
            start = i - static_cast<IndexType>(vector_size);
        }

        for (j=start; j < i; ++j)
        {
            // Projecting i-th vector to j-th vector
            inner_prod = cuVectorOperations<DataType>::inner_product(
                    cublas_handle, &vectors[i*vector_size],
                    &vectors[j*vector_size], vector_size);

            // Norm of the j-th vector
            norm = cuVectorOperations<DataType>::euclidean_norm(
                    cublas_handle, &vectors[j*vector_size], vector_size);

            // Subtraction
            DataType scale = inner_prod / (norm * norm);
            cuVectorOperations<DataType>::subtract_scaled_vector(
                    cublas_handle, &vectors[vector_size*j], vector_size, scale,
                    &vectors[vector_size*i]);
        }
    }
}


// ===============================
// Explicit template instantiation
// ===============================

template class cuOrthogonalization<float>;
template class cuOrthogonalization<double>;
