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

#include "./cu_vector_operations.h"
#include <cmath>  // sqrt
#include <cassert>  // assert
#include "./cublas_interface.h"  // cublas_interface


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
void cuVectorOperations<DataType>::copy_vector(
        cublasHandle_t cublas_handle,
        const DataType* input_vector,
        const LongIndexType vector_size,
        DataType* output_vector)
{
    int incx = 1;
    int incy = 1;

    cublasStatus_t status = cublas_interface::cublasXcopy(
            cublas_handle, vector_size, input_vector, incx, output_vector,
            incy);

    assert(status == CUBLAS_STATUS_SUCCESS);
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
void cuVectorOperations<DataType>::copy_scaled_vector(
        cublasHandle_t cublas_handle,
        const DataType* input_vector,
        const LongIndexType vector_size,
        const DataType scale,
        DataType* output_vector)
{
    cublasStatus_t status;
    int incx = 1;
    int incy = 1;

    // Copy input to output vector
    status = cublas_interface::cublasXcopy(cublas_handle, vector_size,
                                           input_vector, incx,
                                           output_vector, incy);

    assert(status == CUBLAS_STATUS_SUCCESS);

    // Scale outpu vector
    status = cublas_interface::cublasXscal(cublas_handle, vector_size, &scale,
                                           output_vector, incy);

    assert(status == CUBLAS_STATUS_SUCCESS);
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
void cuVectorOperations<DataType>::subtract_scaled_vector(
        cublasHandle_t cublas_handle,
        const DataType* input_vector,
        const LongIndexType vector_size,
        const DataType scale,
        DataType* output_vector)
{
    if (scale == 0.0)
    {
        return;
    }

    int incx = 1;
    int incy = 1;

    DataType neg_scale = -scale;
    cublasStatus_t status = cublas_interface::cublasXaxpy(
            cublas_handle, vector_size, &neg_scale, input_vector, incx,
            output_vector, incy);

    assert(status == CUBLAS_STATUS_SUCCESS);
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
DataType cuVectorOperations<DataType>::inner_product(
        cublasHandle_t cublas_handle,
        const DataType* vector1,
        const DataType* vector2,
        const LongIndexType vector_size)
{
    DataType inner_prod;
    int incx = 1;
    int incy = 1;

    cublasStatus_t status = cublas_interface::cublasXdot(
            cublas_handle, vector_size, vector1, incx, vector2, incy,
            &inner_prod);

    assert(status == CUBLAS_STATUS_SUCCESS);

    return inner_prod;
}


// ==============
// euclidean norm
// ==============

/// \brief     Computes the Euclidean 2-norm of a 1D array.
///
/// \param[in] vector
///            A pointer to 1D array
/// \param[in] vector_size
///            Length of the array
/// \return    Euclidean norm

template <typename DataType>
DataType cuVectorOperations<DataType>::euclidean_norm(
        cublasHandle_t cublas_handle,
        const DataType* vector,
        const LongIndexType vector_size)
{
    DataType norm;
    int incx = 1;

    cublasStatus_t status = cublas_interface::cublasXnrm2(
            cublas_handle, vector_size, vector, incx, &norm);

    assert(status == CUBLAS_STATUS_SUCCESS);

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
DataType cuVectorOperations<DataType>::normalize_vector_in_place(
        cublasHandle_t cublas_handle,
        DataType* vector,
        const LongIndexType vector_size)
{
    // Norm of vector
    DataType norm = cuVectorOperations<DataType>::euclidean_norm(
            cublas_handle, vector, vector_size);

    // Normalize in place
    DataType scale = 1.0 / norm;
    int incx = 1;
    cublasStatus_t status = cublas_interface::cublasXscal(
            cublas_handle, vector_size, &scale, vector, incx);

    assert(status == CUBLAS_STATUS_SUCCESS);

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
DataType cuVectorOperations<DataType>::normalize_vector_and_copy(
        cublasHandle_t cublas_handle,
        const DataType* vector,
        const LongIndexType vector_size,
        DataType* output_vector)
{
    // Norm of vector
    DataType norm = cuVectorOperations<DataType>::euclidean_norm(
            cublas_handle, vector, vector_size);

    // Normalize to output
    DataType scale = 1.0 / norm;
    cuVectorOperations<DataType>::copy_scaled_vector(cublas_handle, vector,
                                                     vector_size, scale,
                                                     output_vector);

    return norm;
}


// ===============================
// Explicit template instantiation
// ===============================

template class cuVectorOperations<float>;
template class cuVectorOperations<double>;
