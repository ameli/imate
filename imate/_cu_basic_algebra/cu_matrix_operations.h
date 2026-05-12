/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */


#ifndef _CU_BASIC_ALGEBRA_CU_MATRIX_OPERATIONS_H_
#define _CU_BASIC_ALGEBRA_CU_MATRIX_OPERATIONS_H_

// =======
// Headers
// =======

#include <cusparse.h>  // cusparseHandle_t
#include "../_definitions/types.h"  // IndexType, LongIndexType, FlagType

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
// Matrix Operations
// =================

/// \class cuMatrixOperations
///
/// \brief   A static class for matrix-vector operations, which are similar to
///          the level-2 operations of the BLAS library. This class acts as a
///          templated namespace, where all member methods are *public* and
///          *static*.
///
/// \details This class implements matrix-vector multiplication for three types
///          of matrices:
///
///          * Dense matrix (both row major and column major)
///          * Compressed sparse row matrix (CSR)
///          * Compressed sparse column matrix (CSC)
///
///          For each of the above matrix types, there are four kinds of matrix
///          vector multiplications implemented.
///
///          1. \c dot : performs \f$ \boldsymbol{c} = \mathbf{A}
///             \boldsymbol{b} \f$.
///          2. \c dot_plus : performs \f$ \boldsymbol{c} = \boldsymbol{c} +
///             \alpha \mathbf{A} \boldsymbol{b} \f$.
///          3. \c transpose_dot : performs \f$ \boldsymbol{c} =
///             \mathbf{A}^{\intercal} \boldsymbol{b} \f$.
///          4. \c transpose_dot_plus : performs \f$ \boldsymbol{c} =
///             \boldsymbol{c} + \alpha \mathbf{A}^{\intercal} \boldsymbol{b}
///             \f$.
///
/// \sa      cuVectorOperations

template <typename DataType>
class cuMatrixOperations
{
    public:

        // dense matvec
        static void dense_matvec(
                cublasHandle_t cublas_handle,
                const DataType* RESTRICT A,
                const DataType* RESTRICT b,
                const LongIndexType num_rows,
                const LongIndexType num_columns,
                const FlagType A_is_row_major,
                DataType* RESTRICT c);

        // dense matvec plus
        static void dense_matvec_plus(
                cublasHandle_t cublas_handle,
                const DataType* RESTRICT A,
                const DataType* RESTRICT b,
                const DataType alpha,
                const LongIndexType num_rows,
                const LongIndexType num_columns,
                const FlagType A_is_row_major,
                DataType* RESTRICT c);

        // dense transposed matvec
        static void dense_transposed_matvec(
                cublasHandle_t cublas_handle,
                const DataType* RESTRICT A,
                const DataType* RESTRICT b,
                const LongIndexType num_rows,
                const LongIndexType num_columns,
                const FlagType A_is_row_major,
                DataType* RESTRICT c);

        // dense transposed matvec plus
        static void dense_transposed_matvec_plus(
                cublasHandle_t cublas_handle,
                const DataType* RESTRICT A,
                const DataType* RESTRICT b,
                const DataType alpha,
                const LongIndexType num_rows,
                const LongIndexType num_columns,
                const FlagType A_is_row_major,
                DataType* RESTRICT c);

        // CSR matvec
        static void csr_matvec(
                cusparseHandle_t cusparse_handle,
                const DataType* RESTRICT A_data,
                const LongIndexType* RESTRICT A_column_indices,
                const LongIndexType* RESTRICT A_index_pointer,
                const DataType* RESTRICT b,
                const LongIndexType num_rows,
                DataType* RESTRICT c);

        // CSR matvec plus
        static void csr_matvec_plus(
                cusparseHandle_t cusparse_handle,
                const DataType* RESTRICT A_data,
                const LongIndexType* RESTRICT A_column_indices,
                const LongIndexType* RESTRICT A_index_pointer,
                const DataType* RESTRICT b,
                const DataType alpha,
                const LongIndexType num_rows,
                DataType* RESTRICT c);

        // CSR transposed matvec
        static void csr_transposed_matvec(
                cusparseHandle_t cusparse_handle,
                const DataType* RESTRICT A_data,
                const LongIndexType* RESTRICT A_column_indices,
                const LongIndexType* RESTRICT A_index_pointer,
                const DataType* RESTRICT b,
                const LongIndexType num_rows,
                const LongIndexType num_columns,
                DataType* RESTRICT c);

        // CSR transposed matvec plus
        static void csr_transposed_matvec_plus(
                cusparseHandle_t cusparse_handle,
                const DataType* RESTRICT A_data,
                const LongIndexType* RESTRICT A_column_indices,
                const LongIndexType* RESTRICT A_index_pointer,
                const DataType* RESTRICT b,
                const DataType alpha,
                const LongIndexType num_rows,
                const LongIndexType num_columns,
                DataType* RESTRICT c);

        // CSC matvec
        static void csc_matvec(
                cusparseHandle_t cusparse_handle,
                const DataType* RESTRICT A_data,
                const LongIndexType* RESTRICT A_row_indices,
                const LongIndexType* RESTRICT A_index_pointer,
                const DataType* RESTRICT b,
                const LongIndexType num_rows,
                const LongIndexType num_columns,
                DataType* RESTRICT c);

        // CSC matvec plus
        static void csc_matvec_plus(
                cusparseHandle_t cusparse_handle,
                const DataType* RESTRICT A_data,
                const LongIndexType* RESTRICT A_row_indices,
                const LongIndexType* RESTRICT A_index_pointer,
                const DataType* RESTRICT b,
                const DataType alpha,
                const LongIndexType num_rows,
                const LongIndexType num_columns,
                DataType* RESTRICT c);

        // CSC transposed matvec
        static void csc_transposed_matvec(
                cusparseHandle_t cusparse_handle,
                const DataType* RESTRICT A_data,
                const LongIndexType* RESTRICT A_row_indices,
                const LongIndexType* RESTRICT A_index_pointer,
                const DataType* RESTRICT b,
                const LongIndexType num_columns,
                DataType* RESTRICT c);

        // CSC transposed matvec plus
        static void csc_transposed_matvec_plus(
                cusparseHandle_t cusparse_handle,
                const DataType* RESTRICT A_data,
                const LongIndexType* RESTRICT A_row_indices,
                const LongIndexType* RESTRICT A_index_pointer,
                const DataType* RESTRICT b,
                const DataType alpha,
                const LongIndexType num_columns,
                DataType* RESTRICT c);

        // Create Band Matrix
        static void create_band_matrix(
                cusparseHandle_t cublas_handle,
                const DataType* RESTRICT diagonals,
                const DataType* RESTRICT supdiagonals,
                const IndexType non_zero_size,
                const FlagType tridiagonal,
                DataType** matrix);
};

#endif  // _CU_BASIC_ALGEBRA_CU_MATRIX_OPERATIONS_H_
