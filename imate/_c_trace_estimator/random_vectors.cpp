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
// Headers
// =======

#include "./random_vectors.h"
#include <omp.h>  // omp_set_num_threads
#include <ctime>  // time, time_t
#include <cstdlib>  // rand, srand, RAND_MAX
#include <cmath>  // copysign
#include "./c_orthogonalization.h"  // cOrthogonalization
#include "../_c_basic_algebra/c_vector_operations.h"  // cVectorOperations


// ==============================
// generate random column vectors
// ==============================

/// \brief         Generates a set of random column vectors using Rademacher
///                distribution. The column vectors are normalized to unit
///                norm. If desired, the vectors can also be orthogonalized by
///                setting \c orthogonalize flag to a non-zero value. The
///                computation is parallelized using OpenMP over the number of
///                column vectors.
///
/// \param[in,out] vectors
///                1D array representing 2D of vectors of size
///                \c (vector_size,num_vectors). This array will be written
///                in-place and serves as the output. Note this is Fortran
///                ordering, meaning that the first index of the equivalent 2D
///                array is contiguous. To refer to the i-th element of the
///                j-th column vector, use \c &vectors[j*vector_size+i].
/// \param[in]     vector_size
///                Number of rows of vectors array.
/// \param[in]     num_vectors
///                Number of columns of vectors array.
/// \param[in]     orthogonalize
///                A flag, when set to a non-zero value, the vectors are
///                orthogonalized using modified Gram-Schmidt process.
///                Otherwise, the generated vectors remain enact.
/// \param[in]     num_threads
///                Number of OpenMP parallel threads

template <typename DataType>
void RandomVectors<DataType>::generate_random_column_vectors(
        DataType* vectors,
        const LongIndexType vector_size,
        const IndexType num_vectors,
        const IndexType orthogonalize,
        const IndexType num_threads)
{
    // Set the seed of rand function
    std::srand(std::time(0));

    // Set the number of threads
    omp_set_num_threads(num_threads);

    // Using max possible chunk size for parallel schedules
    IndexType chunk_size = \
        static_cast<int>((static_cast<DataType>(num_vectors)) / num_threads);
    if (chunk_size < 1)
    {
        chunk_size = 1;
    }

    // Shared-memory parallelism over individual column vectors
    #pragma omp parallel for schedule(static, chunk_size)
    for (IndexType j=0; j < num_vectors; ++j)
    {
        // Inner iteration over the contiguous elements of vector array
        for (LongIndexType i=0; i < vector_size; ++i)
        {
            // Produce Rademacher distribution (+1 and -1 with uniform dist)
            vectors[i + vector_size*j] = copysign(
                    1.0,
                    (2.0*std::rand()/(static_cast<DataType>(RAND_MAX)))-1.0);
        }
    }

    // Orthogonalize (optional). This section is not parallel (must be serial)
    if (orthogonalize)
    {
        cOrthogonalization<DataType>::orthogonalize_vectors(
                vectors, vector_size, num_vectors);
    }

    // Normalize (necessary)
    #pragma omp parallel for schedule(static)
    for (IndexType j=0; j < num_vectors; ++j)
    {
        cVectorOperations<DataType>::normalize_vector_in_place(
                &vectors[vector_size*j], vector_size);
    }
}


// ===========================
// generate random row vectors
// ===========================

/// \brief         Generates a set of random row vectors using Rademacher
///                distribution. The row vectors are normalized to unit norm.
///                The computation is parallelized using OpenMP over the number
///                of vectors.
///
/// \param[in,out] vectors
///                2D array of the shape \c (num_vectors, vector_size), which
///                represents a set of row vectors. This array will be written
///                in-place and serves as the output. To refer to the i-th
///                vector, use \c &vectors[i][0]. Note that this 2D array has C
///                (row major) ordering, meaning that the last index is
///                contiguous.
/// \param[in]     vector_size
///                Number of rows of vectors array.
/// \param[in]     num_vectors
///                Number of columns of vectors array.
/// \param[in]     num_threads
///                Number of OpenMP parallel threads

template <typename DataType>
void RandomVectors<DataType>::generate_random_row_vectors(
        DataType** vectors,
        const LongIndexType vector_size,
        const IndexType num_vectors,
        const IndexType num_threads)
{
    // Set the seed of rand function
    std::srand(std::time(0));

    // Set the number of threads
    omp_set_num_threads(num_threads);

    // Using max possible chunk size for parallel schedules
    IndexType chunk_size = \
        static_cast<int>((static_cast<DataType>(num_vectors)) / num_threads);
    if (chunk_size < 1)
    {
        chunk_size = 1;
    }

    // Shared-memory parallelism over individual row vectors
    #pragma omp parallel for schedule(static, chunk_size)
    for (IndexType i=0; i < num_vectors; ++i)
    {
        // Inner iteration over the contiguous elements of vector array
        for (LongIndexType j=0; j < vector_size; ++j)
        {
            // Produce Rademacher distribution (+1 and -1 with uniform dist)
            vectors[i][j] = copysign(
                    1.0,
                    (2.0*std::rand()/(static_cast<DataType>(RAND_MAX)))-1.0);
        }

        // Normalize
        cVectorOperations<DataType>::normalize_vector_in_place(
                &vectors[i][0], vector_size);
    }
}


// ===============================
// Explicit template instantiation
// ===============================

template class RandomVectors<float>;
template class RandomVectors<double>;
template class RandomVectors<long double>;
