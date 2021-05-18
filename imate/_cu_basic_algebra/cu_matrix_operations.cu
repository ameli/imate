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
#include "./cublas_interface.h"  // cublas_interface
#include "./cusparse_interface.h"  // cusparse_interface


// ============
// dense matvec
// ============

/// \brief      Computes the matrix vector multiplication \f$ \boldsymbol{c} =
///             \mathbf{A} \boldsymbol{b} \f$ where \f$ \mathbf{A} \f$ is a
///             dense matrix.
///
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
        const DataType* A,
        const DataType* b,
        const LongIndexType num_rows,
        const LongIndexType num_columns,
        const FlagType A_is_row_major,
        DataType* c)
{
    cublasOperation_t trans;
    int m;
    int n;
    int lda;
    DataType alpha = 1.0;
    DataType beta = 0.0;
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
    cublasStatus_t status = cublas_interface::cublasXgemv(cublas_handle, trans,
                                                          m, n, &alpha, A, lda,
                                                          b, incb, &beta, c,
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
        const DataType* A,
        const DataType* b,
        const DataType alpha,
        const LongIndexType num_rows,
        const LongIndexType num_columns,
        const FlagType A_is_row_major,
        DataType* c)
{
    cublasOperation_t trans;
    int m;
    int n;
    int lda;
    DataType beta = 1.0;
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
    cublasStatus_t status = cublas_interface::cublasXgemv(cublas_handle, trans,
                                                          m, n, &alpha, A, lda,
                                                          b, incb, &beta, c,
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
        const DataType* A,
        const DataType* b,
        const LongIndexType num_rows,
        const LongIndexType num_columns,
        const FlagType A_is_row_major,
        DataType* c)
{
    cublasOperation_t trans;
    int m;
    int n;
    int lda;
    DataType alpha = 1.0;
    DataType beta = 0.0;
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
    cublasStatus_t status = cublas_interface::cublasXgemv(cublas_handle, trans,
                                                          m, n, &alpha, A, lda,
                                                          b, incb, &beta, c,
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
        const DataType* A,
        const DataType* b,
        const DataType alpha,
        const LongIndexType num_rows,
        const LongIndexType num_columns,
        const FlagType A_is_row_major,
        DataType* c)
{
    if (alpha == 0.0)
    {
        return;
    }

    cublasOperation_t trans;
    int m;
    int n;
    int lda;
    DataType beta = 0.0;
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
    cublasStatus_t status = cublas_interface::cublasXgemv(cublas_handle, trans,
                                                          m, n, &alpha, A, lda,
                                                          b, incb, &beta, c,
                                                          incc);
    assert(status == CUBLAS_STATUS_SUCCESS);
}


// ==========
// csr matvec
// ==========

/// \brief      Computes \f$ \boldsymbol{c} = \mathbf{A} \boldsymbol{b} \f$
///             where \f$ \mathbf{A}\ f$ is compressed sparse row (CSR) matrix
///             and \f$ \boldsymbol{b} \f$ is a dense vector. The output \f$
///             \boldsymbol{c} \f$ is a dense vector.
///
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
        const DataType* A_data,
        const LongIndexType* A_column_indices,
        const LongIndexType* A_index_pointer,
        const DataType* b,
        const LongIndexType num_rows,
        DataType* c)
{
    LongIndexType index_pointer;
    LongIndexType row;
    LongIndexType column;
    DataType sum;

    for (row=0; row < num_rows; ++row)
    {
        sum = 0.0;
        for (index_pointer=A_index_pointer[row];
             index_pointer < A_index_pointer[row+1];
             ++index_pointer)
        {
            column = A_column_indices[index_pointer];
            sum += A_data[index_pointer] * b[column];
        }
        c[row] = sum;
    }
}


// ===============
// csr matvec plus
// ===============

/// \brief         Computes \f$ \boldsymbol{c} = \boldsymbol{c} + \alpha
///                \mathbf{A} \boldsymbol{b} \f$ where \f$ \mathbf{A}\ f$ is
///                compressed sparse row (CSR) matrix and \f$ \boldsymbol{b}
///                \f$ is a dense vector. The output \f$ \boldsymbol{c} \f$ is
///                a dense vector.
///
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
        const DataType* A_data,
        const LongIndexType* A_column_indices,
        const LongIndexType* A_index_pointer,
        const DataType* b,
        const DataType alpha,
        const LongIndexType num_rows,
        DataType* c)
{
    if (alpha == 0.0)
    {
        return;
    }

    LongIndexType index_pointer;
    LongIndexType row;
    LongIndexType column;
    DataType sum;

    for (row=0; row < num_rows; ++row)
    {
        sum = 0.0;
        for (index_pointer=A_index_pointer[row];
             index_pointer < A_index_pointer[row+1];
             ++index_pointer)
        {
            column = A_column_indices[index_pointer];
            sum += A_data[index_pointer] * b[column];
        }
        c[row] += alpha * sum;
    }
}


// =====================
// csr transposed matvec
// =====================

/// \brief      Computes \f$\boldsymbol{c} =\mathbf{A}^{\intercal}
///             \boldsymbol{b}\f$ where \f$ \mathbf{A} \f$ is compressed sparse
///             row (CSR) matrix and \f$ \boldsymbol{b} \f$ is a dense vector.
///             The output \f$ \boldsymbol{c} \f$ is a dense vector.
///
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
        const DataType* A_data,
        const LongIndexType* A_column_indices,
        const LongIndexType* A_index_pointer,
        const DataType* b,
        const LongIndexType num_rows,
        const LongIndexType num_columns,
        DataType* c)
{
    LongIndexType index_pointer;
    LongIndexType row;
    LongIndexType column;

    // Initialize output to zero
    for (column=0; column < num_columns; ++column)
    {
        c[column] = 0.0;
    }

    for (row=0; row < num_rows; ++row)
    {
        for (index_pointer=A_index_pointer[row];
             index_pointer < A_index_pointer[row+1];
             ++index_pointer)
        {
            column = A_column_indices[index_pointer];
            c[column] += A_data[index_pointer] * b[row];
        }
    }
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
        const DataType* A_data,
        const LongIndexType* A_column_indices,
        const LongIndexType* A_index_pointer,
        const DataType* b,
        const DataType alpha,
        const LongIndexType num_rows,
        const LongIndexType num_columns,
        DataType* c)
{
    if (alpha == 0.0)
    {
        return;
    }

    LongIndexType index_pointer;
    LongIndexType row;
    LongIndexType column;

    for (row=0; row < num_rows; ++row)
    {
        for (index_pointer=A_index_pointer[row];
             index_pointer < A_index_pointer[row+1];
             ++index_pointer)
        {
            column = A_column_indices[index_pointer];
            c[column] += alpha * A_data[index_pointer] * b[row];
        }
    }
}


// ==========
// csc matvec
// ==========

/// \brief      Computes \f$ \boldsymbol{c} = \mathbf{A} \boldsymbol{b} \f$
///             where \f$ \mathbf{A} \f$ is compressed sparse column (CSC)
///             matrix and \f$ \boldsymbol{b} \f$ is a dense vector. The output
///             \f$ \boldsymbol{c} \f$ is a dense vector.
///
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
        const DataType* A_data,
        const LongIndexType* A_row_indices,
        const LongIndexType* A_index_pointer,
        const DataType* b,
        const LongIndexType num_rows,
        const LongIndexType num_columns,
        DataType* c)
{
    LongIndexType index_pointer;
    LongIndexType row;
    LongIndexType column;

    // Initialize output to zero
    for (row=0; row < num_rows; ++row)
    {
        c[row] = 0.0;
    }

    for (column=0; column < num_columns; ++column)
    {
        for (index_pointer=A_index_pointer[column];
             index_pointer < A_index_pointer[column+1];
             ++index_pointer)
        {
            row = A_row_indices[index_pointer];
            c[row] += A_data[index_pointer] * b[column];
        }
    }
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
        const DataType* A_data,
        const LongIndexType* A_row_indices,
        const LongIndexType* A_index_pointer,
        const DataType* b,
        const DataType alpha,
        const LongIndexType num_rows,
        const LongIndexType num_columns,
        DataType* c)
{
    if (alpha == 0.0)
    {
        return;
    }

    LongIndexType index_pointer;
    LongIndexType row;
    LongIndexType column;

    for (column=0; column < num_columns; ++column)
    {
        for (index_pointer=A_index_pointer[column];
             index_pointer < A_index_pointer[column+1];
             ++index_pointer)
        {
            row = A_row_indices[index_pointer];
            c[row] += alpha * A_data[index_pointer] * b[column];
        }
    }
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
        const DataType* A_data,
        const LongIndexType* A_row_indices,
        const LongIndexType* A_index_pointer,
        const DataType* b,
        const LongIndexType num_columns,
        DataType* c)
{
    LongIndexType index_pointer;
    LongIndexType row;
    LongIndexType column;
    DataType sum;

    for (column=0; column < num_columns; ++column)
    {
        sum = 0.0;
        for (index_pointer=A_index_pointer[column];
             index_pointer < A_index_pointer[column+1];
             ++index_pointer)
        {
            row = A_row_indices[index_pointer];
            sum += A_data[index_pointer] * b[row];
        }
        c[column] = sum;
    }
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
        const DataType* A_data,
        const LongIndexType* A_row_indices,
        const LongIndexType* A_index_pointer,
        const DataType* b,
        const DataType alpha,
        const LongIndexType num_columns,
        DataType* c)
{
    if (alpha == 0.0)
    {
        return;
    }

    LongIndexType index_pointer;
    LongIndexType row;
    LongIndexType column;
    DataType sum;

    for (column=0; column < num_columns; ++column)
    {
        sum = 0.0;
        for (index_pointer=A_index_pointer[column];
             index_pointer < A_index_pointer[column+1];
             ++index_pointer)
        {
            row = A_row_indices[index_pointer];
            sum += A_data[index_pointer] * b[row];
        }
        c[column] += alpha * sum;
    }
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
        cusparseHandle_t cusparse_handle,
        const DataType* diagonals,
        const DataType* supdiagonals,
        const IndexType non_zero_size,
        const FlagType tridiagonal,
        DataType** matrix)
{
    for (IndexType j=0; j < non_zero_size; ++j)
    {
        // Diagonals
        matrix[j][j] = diagonals[j];

        // Off diagonals
        if (j < non_zero_size-1)
        {
            // Sup-diagonal
            matrix[j][j+1] = supdiagonals[j];

            // Sub-diagonal, making symmetric tri-diagonal matrix
            if (tridiagonal)
            {
                matrix[j+1][j] = supdiagonals[j];
            }
        }
    }
}


// ===============================
// Explicit template instantiation
// ===============================

template class cuMatrixOperations<float>;
template class cuMatrixOperations<double>;
