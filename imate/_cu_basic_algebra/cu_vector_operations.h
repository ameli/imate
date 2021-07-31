/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */


#ifndef _CU_BASIC_ALGEBRA_CU_VECTOR_OPERATIONS_H_
#define _CU_BASIC_ALGEBRA_CU_VECTOR_OPERATIONS_H_

// =======
// Headers
// =======

#include <cublas_v2.h>
#include "../_definitions/types.h"  // LongIndexType


// =================
// Vector Operations
// =================

/// \class cuVectorOperations
///
/// \brief A static class for vector operations, similar to level-1 operations
///        of the BLAS library. This class acts as a templated namespace, where
///        all member methods are *public* and *static*.
///
/// \sa    MatrixOperations

template <typename DataType>
class cuVectorOperations
{
    public:

        // copy vector
        static void copy_vector(
                cublasHandle_t cublas_handle,
                const DataType* input_vector,
                const LongIndexType vector_size,
                DataType* output_vector);

        // copy scaled vector
        static void copy_scaled_vector(
                cublasHandle_t cublas_handle,
                const DataType* input_vector,
                const LongIndexType vector_size,
                const DataType scale,
                DataType* output_vector);

        // subtract scaled vector
        static void subtract_scaled_vector(
                cublasHandle_t cublas_handle,
                const DataType* input_vector,
                const LongIndexType vector_size,
                const DataType scale,
                DataType* output_vector);

        // inner product
        static DataType inner_product(
                cublasHandle_t cublas_handle,
                const DataType* vector1,
                const DataType* vector2,
                const LongIndexType vector_size);

        // euclidean norm
        static DataType euclidean_norm(
                cublasHandle_t cublas_handle,
                const DataType* vector,
                const LongIndexType vector_size);

        // normalize vector in place
        static DataType normalize_vector_in_place(
                cublasHandle_t cublas_handle,
                DataType* vector,
                const LongIndexType vector_size);

        // normalize vector and copy
        static DataType normalize_vector_and_copy(
                cublasHandle_t cublas_handle,
                const DataType* vector,
                const LongIndexType vector_size,
                DataType* output_vector);
};

#endif  // _CU_BASIC_ALGEBRA_CU_VECTOR_OPERATIONS_H_
