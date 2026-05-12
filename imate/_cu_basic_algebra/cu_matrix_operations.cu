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

#include "./cu_matrix_operations.h"
#include <cassert>  // assert
#include <omp.h>  // omp_in_parallel
#include "../_cu_definitions/cu_types.h" // __nv_fp8_e5m2, __nv_fp8_e4m3,
                                         // __half, __nv_bfloat16
#include "../_cu_arithmetics/cu_arithmetics.h"  // cu_arithmetics
#include "./cublas_api.h"  // cublas_api
#include "./cusparse_api.h"  // cusparse_api
#include "../_definitions/definitions.h"  // LARGE_ARRAY_SIZE
#include <stdexcept>  // std::invalid_argument


// ============
// dense matvec
// ============

/// \brief      Computes the matrix vector multiplication \f$ \boldsymbol{c} =
///             \mathbf{A} \boldsymbol{b} \f$ where \f$ \mathbf{A} \f$ is a
///             dense matrix.
///
/// \param[in]  cublas_handle
///             The cuBLAS object handle.
/// \param[in]  A
///             1D array that represents a 2D dense array with either C (row)
///             major ordering or Fortran (column) major ordering. The major
///             ordering should de defined by \c A_is_row_major flag.
/// \param[in]  b
///             Column vector
/// \param[in]  num_rows
///             Number of rows of \c A
/// \param[in]  num_columns
///             Number of columns of \c A
/// \param[in]  A_is_row_major
///             Boolean, can be \c 0 or \c 1 as follows:
///             * If \c A is row major (C ordering where the last index is
///               contiguous) this value should be \c 1.
///             * If \c A is column major (Fortran ordering where the first
///               index is contiguous), this value should be set to \c 0.
/// \param[out] c
///             The output column vector (written in-place).

template <typename DataType>
void cuMatrixOperations<DataType>::dense_matvec(
        cublasHandle_t cublas_handle,
        const DataType* RESTRICT A,
        const DataType* RESTRICT b,
        const LongIndexType num_rows,
        const LongIndexType num_columns,
        const FlagType A_is_row_major,
        DataType* RESTRICT c)
{
    cublasOperation_t trans;
    int m;
    int n;
    int lda;
    DataType alpha = cu_arithmetics::cast<float, DataType>(1.0f);
    DataType beta = cu_arithmetics::cast<float, DataType>(0.0f);
    int incb = 1;
    int incc = 1;

    // Since cublas accepts column major (Fortran) ordering, use transpose for
    // row_major matrix.
    if (A_is_row_major)
    {
        // A is row-major, not compatible with cublas. Use transpose instead.
        trans = CUBLAS_OP_T;
        m = num_columns;
        n = num_rows;
    }
    else
    {
        // A is column-major, compatible with cublas.
        trans = CUBLAS_OP_N;
        m = num_rows;
        n = num_columns;
    }

    lda = m;

    // Calling cublas
    cublasStatus_t status = cublas_api::cublasXgemv<DataType>(
            cublas_handle, trans, m, n, &alpha, A, lda, b, incb, &beta, c,
            incc);

    assert(status == CUBLAS_STATUS_SUCCESS);
}


// =================
// dense matvec plus
// =================

/// \brief         Computes the operation \f$ \boldsymbol{c} = \boldsymbol{c} +
///                \alpha \mathbf{A} \boldsymbol{b} \f$ where \f$ \mathbf{A}
///                \f$ is a dense matrix.
///
/// \param[in]     cublas_handle
///                The cuBLAS object handle.
/// \param[in]     A
///                1D array that represents a 2D dense array with either C
///                (row) major ordering or Fortran (column) major ordering. The
///                major ordering should de defined by \c A_is_row_major flag.
/// \param[in]     b
///                Column vector
/// \param[in]     alpha
///                A scalar that scales the matrix vector multiplication.
/// \param[in]     num_rows
///                Number of rows of \c A
/// \param[in]     num_columns
///                Number of columns of \c A
/// \param[in]     A_is_row_major
///                Boolean, can be \c 0 or \c 1 as follows:
///                * If \c A is row major (C ordering where the last index is
///                  contiguous) this value should be \c 1.
///                * If \c A is column major (Fortran ordering where the first
///                  index is contiguous), this value should be set to \c 0.
/// \param[in,out] c
///                The output column vector (written in-place).

template <typename DataType>
void cuMatrixOperations<DataType>::dense_matvec_plus(
        cublasHandle_t cublas_handle,
        const DataType* RESTRICT A,
        const DataType* RESTRICT b,
        const DataType alpha,
        const LongIndexType num_rows,
        const LongIndexType num_columns,
        const FlagType A_is_row_major,
        DataType* RESTRICT c)
{
    DataType zero = cu_arithmetics::cast<float, DataType>(0.0f);
    if (cu_arithmetics::is_equal(alpha, zero))
    {
        return;
    }
    
    cublasOperation_t trans;
    int m;
    int n;
    int lda;
    DataType beta = cu_arithmetics::cast<float, DataType>(1.0f);
    int incb = 1;
    int incc = 1;

    // Since cublas accepts column major (Fortran) ordering, use transpose for
    // row_major matrix.
    if (A_is_row_major)
    {
        trans = CUBLAS_OP_T;
        m = num_columns;
        n = num_rows;
    }
    else
    {
        trans = CUBLAS_OP_N;
        m = num_rows;
        n = num_columns;
    }

    lda = m;

    // Calling cublas
    cublasStatus_t status = cublas_api::cublasXgemv<DataType>(
            cublas_handle, trans, m, n, &alpha, A, lda, b, incb, &beta, c,
            incc);

    assert(status == CUBLAS_STATUS_SUCCESS);
}


// =======================
// dense transposed matvec
// =======================

/// \brief      Computes matrix vector multiplication \f$\boldsymbol{c} =
///             \mathbf{A}^{\intercal} \boldsymbol{b} \f$ where \f$ \mathbf{A}
///             \f$ is dense, and \f$ \mathbf{A}^{\intercal} \f$ is the
///             transpose of the matrix \f$ \mathbf{A} \f$.
///
/// \param[in]  cublas_handle
///             The cuBLAS object handle.
/// \param[in]  A
///             1D array that represents a 2D dense array with either C (row)
///             major ordering or Fortran (column) major ordering. The major
///             ordering should de defined by \c A_is_row_major flag.
/// \param[in]  b
///             Column vector
/// \param[in]  num_rows
///             Number of rows of \c A
/// \param[in]  num_columns
///             Number of columns of \c A
/// \param[in]  A_is_row_major
///             Boolean, can be \c 0 or \c 1 as follows:
///             * If \c A is row major (C ordering where the last index is
///               contiguous) this value should be \c 1.
///             * f \c A is column major (Fortran ordering where the first
///               index is contiguous), this value should be set to \c 0.
/// \param[out] c
///             The output column vector (written in-place).

template <typename DataType>
void cuMatrixOperations<DataType>::dense_transposed_matvec(
        cublasHandle_t cublas_handle,
        const DataType* RESTRICT A,
        const DataType* RESTRICT b,
        const LongIndexType num_rows,
        const LongIndexType num_columns,
        const FlagType A_is_row_major,
        DataType* RESTRICT c)
{
    cublasOperation_t trans;
    int m;
    int n;
    int lda;
    DataType alpha = cu_arithmetics::cast<float, DataType>(1.0f);
    DataType beta = cu_arithmetics::cast<float, DataType>(0.0f);
    int incb = 1;
    int incc = 1;

    // Since cublas accepts column major (Fortran) ordering, use non-transpose
    // for row_major matrix.
    if (A_is_row_major)
    {
        trans = CUBLAS_OP_N;
        m = num_columns;
        n = num_rows;
    }
    else
    {
        trans = CUBLAS_OP_T;
        m = num_rows;
        n = num_columns;
    }

    lda = m;

    // Calling cublas
    cublasStatus_t status = cublas_api::cublasXgemv<DataType>(
            cublas_handle, trans, m, n, &alpha, A, lda, b, incb, &beta, c,
            incc);

    assert(status == CUBLAS_STATUS_SUCCESS);
}


// ============================
// dense transposed matvec plus
// ============================

/// \brief         Computes \f$ \boldsymbol{c} = \boldsymbol{c} + \alpha
///                \mathbf{A}^{\intercal} \boldsymbol{b} \f$ where \f$
///                \mathbf{A} \f$ is dense, and \f$ \mathbf{A}^{\intercal} \f$
///                is the transpose of the matrix \f$ \mathbf{A} \f$.
///
/// \param[in]     cublas_handle
///                The cuBLAS object handle.
/// \param[in]     A
///                1D array that represents a 2D dense array with either C
///                (row) major ordering or Fortran (column) major ordering. The
///                major ordering should de defined by \c A_is_row_major flag.
/// \param[in]     b
///                Column vector
/// \param[in]     alpha
///                A scalar that scales the matrix vector multiplication.
/// \param[in]     num_rows
///                Number of rows of \c A
/// \param[in]     num_columns
///                Number of columns of \c A
/// \param[in]     A_is_row_major
///                Boolean, can be \c 0 or \c 1 as follows:
///                * If \c A is row major (C ordering where the last index is
///                  contiguous) this value should be \c 1.
///                * f \c A is column major (Fortran ordering where the first
///                  index is contiguous), this value should be set to \c 0.
/// \param[in,out] c
///                The output column vector (written in-place).

template <typename DataType>
void cuMatrixOperations<DataType>::dense_transposed_matvec_plus(
        cublasHandle_t cublas_handle,
        const DataType* RESTRICT A,
        const DataType* RESTRICT b,
        const DataType alpha,
        const LongIndexType num_rows,
        const LongIndexType num_columns,
        const FlagType A_is_row_major,
        DataType* RESTRICT c)
{
    DataType zero = cu_arithmetics::cast<float, DataType>(0.0f);
    if (cu_arithmetics::is_equal(alpha, zero))
    {
        return;
    }

    cublasOperation_t trans;
    int m;
    int n;
    int lda;
    DataType beta = cu_arithmetics::cast<float, DataType>(0.0f);
    int incb = 1;
    int incc = 1;

    // Since cublas accepts column major (Fortran) ordering, use non-transpose
    // for row_major matrix.
    if (A_is_row_major)
    {
        trans = CUBLAS_OP_N;
        m = num_columns;
        n = num_rows;
    }
    else
    {
        trans = CUBLAS_OP_T;
        m = num_rows;
        n = num_columns;
    }

    lda = m;

    // Calling cublas
    cublasStatus_t status = cublas_api::cublasXgemv<DataType>(
            cublas_handle, trans, m, n, &alpha, A, lda, b, incb, &beta, c,
            incc);

    assert(status == CUBLAS_STATUS_SUCCESS);
}


// ==========
// csr matvec
// ==========

/// \brief      Computes \f$ \boldsymbol{c} = \mathbf{A} \boldsymbol{b} \f$
///             where \f$ \mathbf{A} \f$ is compressed sparse row (CSR) matrix
///             and \f$ \boldsymbol{b} \f$ is a dense vector. The output \f$
///             \boldsymbol{c} \f$ is a dense vector.
///
/// \param[in]  cusparse_handle
///             The cuSparse object handle.
/// \param[in]  A_data
///             CSR format data array of the sparse matrix. The length of this
///             array is the nnz of the matrix.
/// \param[in]  A_column_indices
///             CSR format column indices of the sparse matrix. The length of
///             this array is the nnz of the matrix.
/// \param[in]  A_index_pointer
///             CSR format index pointer. The length of this array is one plus
///             the number of rows of the matrix. Also, the first element of
///             this array is \c 0, and the last element is the nnz of the
///             matrix.
/// \param[in]  b
///             Column vector with same size of the number of columns of \c A.
/// \param[in]  num_rows
///             Number of rows of the matrix \c A. This is essentially the size
///             of \c A_index_pointer array minus one.
/// \param[out] c
///             Output column vector with the same size as \c b. This array is
///             written in-place.

template <typename DataType>
void cuMatrixOperations<DataType>::csr_matvec(
        cusparseHandle_t cusparse_handle,
        const DataType* RESTRICT A_data,
        const LongIndexType* RESTRICT A_column_indices,
        const LongIndexType* RESTRICT A_index_pointer,
        const DataType* RESTRICT b,
        const LongIndexType num_rows,
        DataType* RESTRICT c)
{
    throw std::runtime_error("Function not implemented.");
}


// ===============
// csr matvec plus
// ===============

/// \brief         Computes \f$ \boldsymbol{c} = \boldsymbol{c} + \alpha
///                \mathbf{A} \boldsymbol{b} \f$ where \f$ \mathbf{A} \f$ is
///                compressed sparse row (CSR) matrix and \f$ \boldsymbol{b}
///                \f$ is a dense vector. The output \f$ \boldsymbol{c} \f$ is
///                a dense vector.
///
/// \param[in]     cusparse_handle
///                The cuSparse object handle.
/// \param[in]     A_data
///                CSR format data array of the sparse matrix. The length of
///                this array is the nnz of the matrix.
/// \param[in]     A_column_indices
///                CSR format column indices of the sparse matrix. The length
///                of this array is the nnz of the matrix.
/// \param[in]     A_index_pointer
///                CSR format index pointer. The length of this array is one
///                plus the number of rows of the matrix. Also, the first
///                element of this array is \c 0, and the last element is the
///                nnz of the matrix.
/// \param[in]     b
///                Column vector with same size of the number of columns of
///                \c A.
/// \param[in]     alpha
///                A scalar that scales the matrix vector multiplication.
/// \param[in]     num_rows
///                Number of rows of the matrix \c A. This is essentially the
///                size of \c A_index_pointer array minus one.
/// \param[in,out] c
///                Output column vector with the same size as \c b. This array
///                is written in-place.

template <typename DataType>
void cuMatrixOperations<DataType>::csr_matvec_plus(
        cusparseHandle_t cusparse_handle,
        const DataType* RESTRICT A_data,
        const LongIndexType* RESTRICT A_column_indices,
        const LongIndexType* RESTRICT A_index_pointer,
        const DataType* RESTRICT b,
        const DataType alpha,
        const LongIndexType num_rows,
        DataType* RESTRICT c)
{
    throw std::runtime_error("Function not implemented.");
}


// =====================
// csr transposed matvec
// =====================

/// \brief      Computes \f$\boldsymbol{c} =\mathbf{A}^{\intercal}
///             \boldsymbol{b}\f$ where \f$ \mathbf{A} \f$ is compressed sparse
///             row (CSR) matrix and \f$ \boldsymbol{b} \f$ is a dense vector.
///             The output \f$ \boldsymbol{c} \f$ is a dense vector.
///
/// \param[in]  cusparse_handle
///             The cuSparse object handle.
/// \param[in]  A_data
///             CSR format data array of the sparse matrix. The length of this
///             array is the nnz of the matrix.
/// \param[in]  A_column_indices
///             CSR format column indices of the sparse matrix. The length of
///             this array is the nnz of the matrix.
/// \param[in]  A_index_pointer
///             CSR format index pointer. The length of this array is one plus
///             the number of rows of the matrix. Also, the first element of
///             this array is \c 0, and the last element is the nnz of the
///             matrix.
/// \param[in]  b
///             Column vector with same size of the number of columns of \c A.
/// \param[in]  num_rows
///             Number of rows of the matrix \c A. This is essentially the size
///             of \c A_index_pointer array minus one.
/// \param[in]  num_columns
///             Number of columns of the matrix \c A.
/// \param[out] c
///             Output column vector with the same size as \c b. This array is
///             written in-place.

template <typename DataType>
void cuMatrixOperations<DataType>::csr_transposed_matvec(
        cusparseHandle_t cusparse_handle,
        const DataType* RESTRICT A_data,
        const LongIndexType* RESTRICT A_column_indices,
        const LongIndexType* RESTRICT A_index_pointer,
        const DataType* RESTRICT b,
        const LongIndexType num_rows,
        const LongIndexType num_columns,
        DataType* RESTRICT c)
{
    throw std::runtime_error("Function not implemented.");
}


// ==========================
// csr transposed matvec plus
// ==========================

/// \brief         Computes \f$ \boldsymbol{c} = \boldsymbol{c} + \alpha
///                \mathbf{A}^{\intercal} \boldsymbol{b}\f$ where \f$
///                \mathbf{A} \f$ is compressed sparse row (CSR) matrix and \f$
///                \boldsymbol{b} \f$ is a dense vector. The output \f$
///                \boldsymbol{c} \f$ is a dense vector.
///
/// \param[in]     cusparse_handle
///                The cuSparse object handle.
/// \param[in]     A_data
///                CSR format data array of the sparse matrix. The length of
///                this array is the nnz of the matrix.
/// \param[in]     A_column_indices
///                CSR format column indices of the sparse matrix. The length
///                of this array is the nnz of the matrix.
/// \param[in]     A_index_pointer
///                CSR format index pointer. The length of this array is one
///                plus the number of rows of the matrix. Also, the first
///                element of this array is \c 0, and the last element is the
///                nnz of the matrix.
/// \param[in]     b
///                Column vector with same size of the number of columns of
///                \c A.
/// \param[in]     alpha
///                A scalar that scales the matrix vector multiplication.
/// \param[in]     num_rows
///                Number of rows of the matrix \c A. This is essentially the
///                size of \c A_index_pointer array minus one.
/// \param[in]     num_columns
///                Number of columns of the matrix \c A.
/// \param[in,out] c
///                Output column vector with the same size as \c b. This array
///                is written in-place.

template <typename DataType>
void cuMatrixOperations<DataType>::csr_transposed_matvec_plus(
        cusparseHandle_t cusparse_handle,
        const DataType* RESTRICT A_data,
        const LongIndexType* RESTRICT A_column_indices,
        const LongIndexType* RESTRICT A_index_pointer,
        const DataType* RESTRICT b,
        const DataType alpha,
        const LongIndexType num_rows,
        const LongIndexType num_columns,
        DataType* RESTRICT c)
{
    throw std::runtime_error("Function not implemented.");
}


// ==========
// csc matvec
// ==========

/// \brief      Computes \f$ \boldsymbol{c} = \mathbf{A} \boldsymbol{b} \f$
///             where \f$ \mathbf{A} \f$ is compressed sparse column (CSC)
///             matrix and \f$ \boldsymbol{b} \f$ is a dense vector. The output
///             \f$ \boldsymbol{c} \f$ is a dense vector.
///
/// \param[in]  cusparse_handle
///             The cuSparse object handle.
/// \param[in]  A_data
///             CSC format data array of the sparse matrix. The length of this
///             array is the nnz of the matrix.
/// \param[in]  A_row_indices
///             CSC format column indices of the sparse matrix. The length of
///             this array is the nnz of the matrix.
/// \param[in]  A_index_pointer
///             CSC format index pointer. The length of this array is one plus
///             the number of columns of the matrix. Also, the first element of
///             this array is \c 0, and the last element is the nnz of the
///             matrix.
/// \param[in]  b
///             Column vector with same size of the number of columns of \c A.
/// \param[in]  num_rows
///             Number of rows of the matrix \c A.
/// \param[in]  num_columns
///             Number of columns of the matrix \c A. This is essentially the
///             size of \c A_index_pointer array minus one.
/// \param[out] c
///             Output column vector with the same size as \c b. This array is
///             written in-place.

template <typename DataType>
void cuMatrixOperations<DataType>::csc_matvec(
        cusparseHandle_t cusparse_handle,
        const DataType* RESTRICT A_data,
        const LongIndexType* RESTRICT A_row_indices,
        const LongIndexType* RESTRICT A_index_pointer,
        const DataType* RESTRICT b,
        const LongIndexType num_rows,
        const LongIndexType num_columns,
        DataType* RESTRICT c)
{
    throw std::runtime_error("Function not implemented.");
}


// ===============
// csc matvec plus
// ===============

/// \brief         Computes \f$ \boldsymbol{c} = \boldsymbol{c} + \alpha
///                \mathbf{A} \boldsymbol{b} \f$ where \f$ \mathbf{A} \f$ is
///                compressed sparse column (CSC) matrix and \f$ \boldsymbol{b}
///                \f$ is a dense vector. The output \f$ \boldsymbol{c} \f$ is
///                a dense vector.
///
/// \param[in]     cusparse_handle
///                The cuSparse object handle.
/// \param[in]     A_data
///                CSC format data array of the sparse matrix. The length of
///                this array is the nnz of the matrix.
/// \param[in]     A_row_indices
///                CSC format column indices of the sparse matrix. The length
///                of this array is the nnz of the matrix.
/// \param[in]     A_index_pointer
///                CSC format index pointer. The length of this array is one
///                plus the number of columns of the matrix. Also, the first
///                element of this array is \c 0, and the last element is the
///                nnz of the matrix.
/// \param[in]     b
///                Column vector with same size of the number of columns of
///                \c A.
/// \param[in]     alpha
///                A scalar that scales the matrix vector multiplication.
/// \param[in]     num_rows
///                Number of rows of the matrix \c A.
/// \param[in]     num_columns
///                Number of columns of the matrix \c A. This is essentially
///                the size of \c A_index_pointer array minus one.
/// \param[in,out] c
///                Output column vector with the same size as \c b. This array
///                is written in-place.

template <typename DataType>
void cuMatrixOperations<DataType>::csc_matvec_plus(
        cusparseHandle_t cusparse_handle,
        const DataType* RESTRICT A_data,
        const LongIndexType* RESTRICT A_row_indices,
        const LongIndexType* RESTRICT A_index_pointer,
        const DataType* RESTRICT b,
        const DataType alpha,
        const LongIndexType num_rows,
        const LongIndexType num_columns,
        DataType* RESTRICT c)
{
    throw std::runtime_error("Function not implemented.");
}


// =====================
// csc transposed matvec
// =====================

/// \brief      Computes \f$\boldsymbol{c} =\mathbf{A}^{\intercal}
///             \boldsymbol{b} \f$ where \f$ \mathbf{A} \f$ is compressed
///             sparse column (CSC) matrix and \f$ \boldsymbol{b} \f$ is a
///             dense vector. The output \f$ \boldsymbol{c} \f$ is a dense
///             vector.
///
/// \param[in]  cusparse_handle
///             The cuSparse object handle.
/// \param[in]  A_data
///             CSC format data array of the sparse matrix. The length of this
///             array is the nnz of the matrix.
/// \param[in]  A_row_indices
///             CSC format column indices of the sparse matrix. The length of
///             this array is the nnz of the matrix.
/// \param[in]  A_index_pointer
///             CSC format index pointer. The length of this array is one plus
///             the number of columns of the matrix. Also, the first element of
///             this array is \c 0, and the last element is the nnz of the
///             matrix.
/// \param[in]  b
///             Column vector with same size of the number of columns of \c A.
/// \param      num_columns
///             Number of columns of the matrix \c A. This is essentially the
///             size of \c A_index_pointer array minus one.
/// \param[out] c
///             Output column vector with the same size as \c b. This array is
///             written in-place.

template <typename DataType>
void cuMatrixOperations<DataType>::csc_transposed_matvec(
        cusparseHandle_t cusparse_handle,
        const DataType* RESTRICT A_data,
        const LongIndexType* RESTRICT A_row_indices,
        const LongIndexType* RESTRICT A_index_pointer,
        const DataType* RESTRICT b,
        const LongIndexType num_columns,
        DataType* RESTRICT c)
{
    throw std::runtime_error("Function not implemented.");
}


// ==========================
// csc transposed matvec plus
// ==========================

/// \brief         Computes \f$ \boldsymbol{c} = \boldsymbol{c} + \alpha
///                \mathbf{A}^{\intercal} \boldsymbol{b} \f$ where \f$
///                \mathbf{A} \f$ is compressed sparse column (CSC) matrix and
///                \f$ \boldsymbol{b} \f$ is a dense vector. The output \f$
///                \boldsymbol{c} \f$ is a dense vector.
///
/// \param[in]     cusparse_handle
///                The cuSparse object handle.
/// \param[in]     A_data
///                CSC format data array of the sparse matrix. The length of
///                this array is the nnz of the matrix.
/// \param[in]     A_row_indices
///                CSC format column indices of the sparse matrix. The length
///                of this array is the nnz of the matrix.
/// \param[in]     A_index_pointer
///                CSC format index pointer. The length of this array is one
///                plus the number of columns of the matrix. Also, the first
///                element of this array is \c 0, and the last element is the
///                nnz of the matrix.
/// \param[in]     b
///                Column vector with same size of the number of columns of
///                \c A.
/// \param[in]     alpha
///                A scalar that scales the matrix vector multiplication.
/// \param         num_columns
///                Number of columns of the matrix \c A. This is essentially
///                the size of \c A_index_pointer array minus one.
/// \param[in,out] c
///                Output column vector with the same size as \c b. This array
///                is written in-place.

template <typename DataType>
void cuMatrixOperations<DataType>::csc_transposed_matvec_plus(
        cusparseHandle_t cusparse_handle,
        const DataType* RESTRICT A_data,
        const LongIndexType* RESTRICT A_row_indices,
        const LongIndexType* RESTRICT A_index_pointer,
        const DataType* RESTRICT b,
        const DataType alpha,
        const LongIndexType num_columns,
        DataType* RESTRICT c)
{
    throw std::runtime_error("Function not implemented.");
}


// ==================
// create band matrix
// ==================

/// \brief      Creates bi-diagonal or symmetric tri-diagonal matrix from the
///             diagonal array (\c diagonals) and off-diagonal array (\c
///             supdiagonals).
///
/// \details    The output is written in place (in \c matrix). The output is
///             only written up to the \c non_zero_size element, that is: \c
///             matrix[:non_zero_size,:non_zero_size] is filled, and the rest
///             is assumed to be zero.
///
///             Depending on \c tridiagonal, the matrix is upper bi-diagonal or
///             symmetric tri-diagonal.
///
/// \param[in]  cublas_handle
///             The cuSparse object handle.
/// \param[in]  diagonals
///             An array of length \c n. All elements \c diagonals create the
///             diagonals of \c matrix.
/// \param[in]  supdiagonals
///             An array of length \c n. Elements \c supdiagonals[0:-1] create
///             the upper off-diagonal of \c matrix, making \c matrix an upper
///             bi-diagonal matrix. In addition, if \c tridiagonal is set to
///             \c 1, the lower off-diagonal is also created similar to the
///             upper off-diagonal, making \c matrix a symmetric tri-diagonal
///             matrix.
/// \param[in]  non_zero_size
///             Up to the \c matrix[:non_zero_size,:non_zero_size] of \c matrix
///             will be written. At most, \c non_zero_size can be \c n, which
///             is the size of \c diagonals array and the size of the square
///             matrix. If \c non_zero_size is less than \c n, it is due to the
///             fact that either \c diagonals or \c supdiagonals has zero
///             elements after the \c size element (possibly due to early
///             termination of Lanczos iterations method).
/// \param[in]  tridiagonal
///             Boolean. If set to \c 0, the matrix \c T becomes upper
///             bi-diagonal. If set to \c 1, the matrix becomes symmetric
///             tri-diagonal.
/// \param[out] matrix
///             A 2D  matrix (written in place) of the shape \c (n,n). This is
///             the output of this function. This matrix is assumed to be
///             initialized to zero before calling this function.

template <typename DataType>
void cuMatrixOperations<DataType>::create_band_matrix(
        cusparseHandle_t cublas_handle,
        const DataType* RESTRICT diagonals,
        const DataType* RESTRICT supdiagonals,
        const IndexType non_zero_size,
        const FlagType tridiagonal,
        DataType** RESTRICT matrix)
{
    throw std::runtime_error("Function not implemented.");
}


// ===============================
// Explicit template instantiation
// ===============================

#if defined(USE_CUDA_FP8_E5M2) && (USE_CUDA_FP8_E5M2 == 1)
    template class cuMatrixOperations<__nv_fp8_e5m2>;
#endif

#if defined(USE_CUDA_FP8_E4M3) && (USE_CUDA_FP8_E4M3 == 1)
    template class cuMatrixOperations<__nv_fp8_e4m3>;
#endif

#if defined(USE_CUDA_FP16) && (USE_CUDA_FP16 == 1)
    template class cuMatrixOperations<__half>;
#endif

#if defined(USE_CUDA_BF16) && (USE_CUDA_BF16 == 1)
    template class cuMatrixOperations<__nv_bfloat16>;
#endif

#if defined(USE_CUDA_FP32) && (USE_CUDA_FP32 == 1)
    template class cuMatrixOperations<float>;
#endif

#if defined(USE_CUDA_FP64) && (USE_CUDA_FP64 == 1)
    template class cuMatrixOperations<double>;
#endif
