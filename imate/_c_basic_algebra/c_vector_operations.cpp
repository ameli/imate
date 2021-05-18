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

#include "./c_vector_operations.h"
#include <cmath>  // sqrt


// ===========
// copy vector
// ===========

/// \brief      Copies a vector to a new vector. Result is written in-place
///
/// \param[in]  input_vector
///             A 1D array
/// \param[in]  vector_size
///             Length of vector array
/// \param[out] output_vector
///             Output vector (written in place).

template <typename DataType>
void cVectorOperations<DataType>::copy_vector(
        const DataType* input_vector,
        const LongIndexType vector_size,
        DataType* output_vector)
{
    for (LongIndexType i=0; i < vector_size; ++i)
    {
        output_vector[i] = input_vector[i];
    }
}

// ==================
// copy scaled vector
// ==================

/// \brief      Scales a vector and stores to a new vector.
///
/// \param[in]  input_vector
///             A 1D array
/// \param[in]  vector_size
///             Length of vector array
/// \param[in]  scale
///             Scale coefficient to the input vector. If this is equal to one,
///             the function effectively becomes the same as \e copy_vector.
/// \param[out] output_vector
///             Output vector (written in place).

template <typename DataType>
void cVectorOperations<DataType>::copy_scaled_vector(
        const DataType* input_vector,
        const LongIndexType vector_size,
        const DataType scale,
        DataType* output_vector)
{
    for (LongIndexType i=0; i < vector_size; ++i)
    {
        output_vector[i] = scale * input_vector[i];
    }
}


// ======================
// subtract scaled vector
// ======================

/// \brief         Subtracts the scaled input vector from the output vector.
///
/// \details       Performs the following operation:
///                \f[
///                    \boldsymbol{b} = \boldsymbol{b} - c \boldsymbol{a},
///                \f]
///                where
///                * \f$ \boldsymbol{a} \f$ is the input vector,
///                * \f$ c \f$ is a scalar scale to the input vector, and
///                * \f$ \boldsymbol{b} \f$ is the output vector that is
///                written in-place.
///
/// \param[in]     input_vector
///                A 1D array
/// \param[in]     vector_size
///                Length of vector array
/// \param[in]     scale
///                Scale coefficient to the input vector.
/// \param[in,out] output_vector Output vector (written in place).

template <typename DataType>
void cVectorOperations<DataType>::subtract_scaled_vector(
        const DataType* input_vector,
        const LongIndexType vector_size,
        const DataType scale,
        DataType* output_vector)
{
    if (scale == 0.0)
    {
        return;
    }

    for (LongIndexType i=0; i < vector_size; ++i)
    {
        output_vector[i] -= scale * input_vector[i];
    }
}


// =============
// inner product
// =============

/// \brief     Computes Euclidean inner product of two vectors.
///
/// \param[in] vector1
///            1D array
/// \param[in] vector2
///            1D array
/// \param[in] vector_size Length of array
/// \return    Inner product of two vectors.

template <typename DataType>
DataType cVectorOperations<DataType>::inner_product(
        const DataType* vector1,
        const DataType* vector2,
        const LongIndexType vector_size)
{
    DataType inner_prod = 0.0;

    for (LongIndexType i=0; i < vector_size; ++i)
    {
        inner_prod += vector1[i] * vector2[i];
    }

    return inner_prod;
}


// ==============
// euclidean norm
// ==============

/// \brief     Computes the Euclidean norm of a 1D array.
///
/// \param[in] vector
///            A pointer to 1D array
/// \param[in] vector_size
///            Length of the array
/// \return    Euclidean norm

template <typename DataType>
DataType cVectorOperations<DataType>::euclidean_norm(
        const DataType* vector,
        const LongIndexType vector_size)
{
    // Compute norm squared
    DataType norm2 = 0.0;

    for (LongIndexType i=0; i < vector_size; ++i)
    {
        norm2 += vector[i] * vector[i];
    }

    // Norm
    DataType norm = sqrt(norm2);

    return norm;
}


// =========================
// normalize vector in place
// =========================

/// \brief          Normalizes a vector based on Euclidean 2-norm. The result
///                 is written in-place.
///
/// \param[in, out] vector
///                 Input vector to be normalized in-place.
/// \param[in]      vector_size
///                 Length of the input vector
/// \return         2-Norm of the input vector (before normalization)

template <typename DataType>
DataType cVectorOperations<DataType>::normalize_vector_in_place(
        DataType* vector,
        const LongIndexType vector_size)
{
    // Norm of vector
    DataType norm = cVectorOperations<DataType>::euclidean_norm(vector,
                                                                vector_size);

    // Normalize in place
    for (LongIndexType i=0; i < vector_size; ++i)
    {
        vector[i] /= norm;
    }

    return norm;
}


// =========================
// normalize vector and copy
// =========================

/// \brief      Normalizes a vector based on Euclidean 2-norm. The result is
///             written into another vector.
///
/// \param[in]  vector
///             Input vector.
/// \param[in]  vector_size
///             Length of the input vector
/// \param[out] output_vector
///             Output vector, which is the normalization of the input vector.
/// \return     2-norm of the input vector

template <typename DataType>
DataType cVectorOperations<DataType>::normalize_vector_and_copy(
        const DataType* vector,
        const LongIndexType vector_size,
        DataType* output_vector)
{
    // Norm of vector
    DataType norm = cVectorOperations<DataType>::euclidean_norm(vector,
                                                                vector_size);

    // Normalize to output
    for (LongIndexType i=0; i < vector_size; ++i)
    {
        output_vector[i] = vector[i] / norm;
    }

    return norm;
}


// ===============================
// Explicit template instantiation
// ===============================

template class cVectorOperations<float>;
template class cVectorOperations<double>;
template class cVectorOperations<long double>;
