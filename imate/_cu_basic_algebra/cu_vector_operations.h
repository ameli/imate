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

#include "../_definitions/types.h"  // LongIndexType

// Avoid CUBLAS numeration value not handled in switch [-Wswitch-enum] warning
#ifdef _MSC_VER
    #pragma warning(push, 0)  // Suppress all warnings from the followings
    #include <cublas_v2.h>
    #pragma warning(pop)  // Restore previous warning level
#elif defined(__INTEL_LLVM_COMPILER) || defined(__INTEL_COMPILER)
    #pragma warning(push, 0)
    #include <cublas_v2.h>
    #pragma warning(pop)
#elif defined(__GNUC__) || defined(__clang__)
    #pragma GCC diagnostic push
    #pragma GCC diagnostic ignored "-Wswitch-enum"
    #include <cublas_v2.h>
    #pragma GCC diagnostic pop
#else
    #include <cublas_v2.h>
#endif

// Restrict qualifier
#if defined(_MSC_VER)
    #define RESTRICT __restrict
#elif defined(__INTEL_COMPILER)
    #define RESTRICT __restrict
#elif defined(__CUDA__) || defined(__GNUC__) || defined(__clang__)
    #define RESTRICT __restrict__
#else
    #define RESTRICT
#endif


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
                const DataType* RESTRICT input_vector,
                const LongIndexType vector_size,
                DataType* RESTRICT output_vector);

        // copy scaled vector
        static void copy_scaled_vector(
                cublasHandle_t cublas_handle,
                const DataType* RESTRICT input_vector,
                const LongIndexType vector_size,
                const DataType scale,
                DataType* RESTRICT output_vector);

        // subtract scaled vector
        static void subtract_scaled_vector(
                cublasHandle_t cublas_handle,
                const DataType* RESTRICT input_vector,
                const LongIndexType vector_size,
                const DataType scale,
                DataType* RESTRICT output_vector);

        // inner product
        static DataType inner_product(
                cublasHandle_t cublas_handle,
                const DataType* RESTRICT vector1,
                const DataType* RESTRICT vector2,
                const LongIndexType vector_size);

        // euclidean norm
        static DataType euclidean_norm(
                cublasHandle_t cublas_handle,
                const DataType* RESTRICT vector,
                const LongIndexType vector_size);

        // normalize vector in place
        static DataType normalize_vector_in_place(
                cublasHandle_t cublas_handle,
                DataType* RESTRICT vector,
                const LongIndexType vector_size);

        // normalize vector and copy
        static DataType normalize_vector_and_copy(
                cublasHandle_t cublas_handle,
                const DataType* RESTRICT vector,
                const LongIndexType vector_size,
                DataType* RESTRICT output_vector);
};

#endif  // _CU_BASIC_ALGEBRA_CU_VECTOR_OPERATIONS_H_
