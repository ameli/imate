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

#include "./c_matrix_operations.h"
#include "../_c_arithmetics/c_arithmetics.h"  // c_arithmetics
#include "../_definitions/definitions.h"  // USE_OPENMP, USE_ANY_CBLAS,
                                          // USE_LOOP_UNROLLING,
                                          // LARGE_ARRAY_SIZE
#if defined(USE_OPENMP) && (USE_OPENMP == 1)
    #include <omp.h>  // omp_in_parallel
#endif

#if defined(USE_ANY_CBLAS) && (USE_ANY_CBLAS == 1)
    #include "./cblas_api.h"  // cblas_api
#endif


// ============
// dense matvec
// ============

/// \brief      Computes the matrix vector multiplication \f$ \boldsymbol{c} =
///             \mathbf{A} \boldsymbol{b} \f$ where \f$ \mathbf{A} \f$ is a
///             dense matrix.
///
/// \details    The reduction variable (here, \c sum ) is of the type
///             <tt>long double</tt>. This is becase when \c DataType is \c
///             float, the summation loses the precision, especially when the
///             vector size is large. It seems that using <tt>long double</tt>
///             is slightly faster than using \c double. The advantage of using
///             a type with larger bits for the reduction variable is only
///             sensible if the compiler is optimized with \c -O2 or \c -O3
///             flags.
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
/// \param[in]  A_is_symmetric
///             Boolean. If \c A is symmetric, set this value to \c 1,
///             otherwise \c 0.
/// \param[out] c
///             The output column vector (written in-place).

template <typename DataType>
void cMatrixOperations<DataType>::dense_matvec(
        const DataType* RESTRICT A,
        const DataType* RESTRICT b,
        const LongIndexType num_rows,
        const LongIndexType num_columns,
        const FlagType A_is_row_major,
        const FlagType A_is_symmetric,
        DataType* RESTRICT c)
{
    #if defined(USE_ANY_CBLAS) && (USE_ANY_CBLAS == 1)

    // Using BLAS
    CBLAS_LAYOUT layout;
    CBLAS_UPLO uplo;
    CBLAS_TRANSPOSE transpose = CblasNoTrans;
    int lda;
    if (A_is_row_major)
    {
        layout = CblasRowMajor;
        uplo = CblasUpper;
        lda = num_columns;
    }
    else
    {
        layout = CblasColMajor;
        uplo = CblasLower;
        lda = num_rows;

        // For efficiency, use transpose op for symmetric column-major matrices
        if (A_is_symmetric)
        {
            transpose = CblasTrans;
        }
    }

    int incb = 1;
    int incc = 1;
    DataType alpha = 1.0;
    DataType beta = 0.0;

    if (A_is_symmetric)
    {
        cblas_api::xsymv(layout, uplo, num_columns, alpha, A, lda, b, incb,
                         beta, c, incc);
    }
    else
    {
        cblas_api::xgemv(layout, transpose, num_rows, num_columns, alpha, A,
                         lda, b, incb, beta, c, incc);
    }

    #else

    // Not using BLAS
    long int j;  // This should be long int to avoid multiplication overflow
    long int jump;
    long double sum;
    const long int chunk = 5;
    const long int num_columns_chunked = num_columns - (num_columns % chunk);

    // Void unused variables to avoid compiler warnings (-Wno-unused-parameter)
    #if !defined(USE_LOOP_UNROLLING) || (USE_LOOP_UNROLLING != 1)
    (void) num_columns_chunked;
    (void) chunk;
    #endif

    // Determine major order of A
    if (A_is_row_major)
    {
        // For row-major (C ordering) matrix A (symmetric or non-symmetric)
        #if defined(USE_OPENMP) && (USE_OPENMP == 1)
        #pragma omp parallel for \
            schedule(static) \
            if (!omp_in_parallel()) \
            default(none) \
            shared(A, b, c, jump, num_rows, num_columns, num_columns_chunked, \
                   chunk) \
            private(j, sum)
        #endif
        for (long int i=0; i < num_rows; ++i)
        {
            sum = 0.0;
            jump = i * num_columns;
            #if defined(USE_LOOP_UNROLLING) && (USE_LOOP_UNROLLING == 1)
            for (j=0; j < num_columns_chunked; j+= chunk)
            {
                // Loop unrolling
                sum += A[jump + j] * b[j] +
                       A[jump + j+1] * b[j+1] +
                       A[jump + j+2] * b[j+2] +
                       A[jump + j+3] * b[j+3] +
                       A[jump + j+4] * b[j+4];
            }
            #endif

            #if defined(USE_LOOP_UNROLLING) && (USE_LOOP_UNROLLING == 1)
            for (j=num_columns_chunked; j < num_columns; ++j)
            #else
            for (j=0; j < num_columns; ++j)
            #endif
            {
                sum += A[jump + j] * b[j];
            }

            c[i] = static_cast<DataType>(sum);
        }
    }
    else if (A_is_symmetric)
    {
        // A is column-major, but symmetric, so we can use its transposed
        // operation instead, which is more efficient for column-major
        // matrices.
        cMatrixOperations<DataType>::dense_transposed_matvec(
            A, b, num_rows, num_columns, A_is_row_major, A_is_symmetric, c);
    }
    else
    {
        // For column-major (Fortran ordering) non-symmetric matrix A
        #if defined(USE_OPENMP) && (USE_OPENMP == 1)
        #pragma omp parallel for \
            schedule(static) \
            if (!omp_in_parallel()) \
            default(none) \
            shared(A, b, c, num_rows, num_columns) \
            private(j, sum)
        #endif
        for (long int i=0; i < num_rows; ++i)
        {
            sum = 0.0;
            for (j=0; j < num_columns; ++j)
            {
                // Make sure j is long int to avoid overflow in num_rows*j
                sum += A[i + num_rows*j] * b[j];
            }
            c[i] = static_cast<DataType>(sum);
        }
    }

    #endif
}


// =================
// dense matvec plus
// =================

/// \brief         Computes the operation \f$ \boldsymbol{c} = \boldsymbol{c} +
///                \alpha \mathbf{A} \boldsymbol{b} \f$ where \f$ \mathbf{A}
///                \f$ is a dense matrix.
///
/// \details       The reduction variable (here, \c sum ) is of the type
///                <tt>long double</tt>. This is becase when \c DataType is \c
///                float the summation loses the precision, especially when the
///                vector size is large. It seems that using <tt>long double
///                </tt> is slightly faster than using \c double. The advantage
///                of using a type with larger bits for the reduction variable
///                is only sensible if the compiler is optimized with \c -O2 or
///                \c -O3 flags.
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
/// \param[in]     A_is_symmetric
///                Boolean. If \c A is symmetric, set this value to \c 1,
///                otherwise \c 0.
/// \param[in,out] c
///                The output column vector (written in-place).

template <typename DataType>
void cMatrixOperations<DataType>::dense_matvec_plus(
        const DataType* RESTRICT A,
        const DataType* RESTRICT b,
        const DataType alpha,
        const LongIndexType num_rows,
        const LongIndexType num_columns,
        const FlagType A_is_row_major,
        const FlagType A_is_symmetric,
        DataType* RESTRICT c)
{
    #if defined(USE_ANY_CBLAS) && (USE_ANY_CBLAS == 1)

    // Using BLAS
    CBLAS_LAYOUT layout;
    CBLAS_UPLO uplo;
    CBLAS_TRANSPOSE transpose = CblasNoTrans;
    int lda;
    if (A_is_row_major)
    {
        layout = CblasRowMajor;
        uplo = CblasUpper;
        lda = num_columns;
    }
    else
    {
        layout = CblasColMajor;
        uplo = CblasLower;
        lda = num_rows;

        // For efficiency, use transpose op for symmetric column-major matrices
        if (A_is_symmetric)
        {
            transpose = CblasTrans;
        }
    }

    int incb = 1;
    int incc = 1;
    DataType beta = 1.0;

    if (A_is_symmetric)
    {
        cblas_api::xsymv(layout, uplo, num_columns, alpha, A, lda, b, incb,
                         beta, c, incc);
    }
    else
    {
        cblas_api::xgemv(layout, transpose, num_rows, num_columns, alpha, A,
                         lda, b, incb, beta, c, incc);
    }

    #else

    // Not using BLAS
    DataType zero = 0.0;
    if (c_arithmetics::is_equal(alpha, zero))
    {
        return;
    }

    long int j;
    long int jump;
    long double sum;
    const long int chunk = 5;
    const long int num_columns_chunked = num_columns - (num_columns % chunk);

    // Void unused variables to avoid compiler warnings (-Wno-unused-parameter)
    #if !defined(USE_LOOP_UNROLLING) || (USE_LOOP_UNROLLING != 1)
    (void) num_columns_chunked;
    (void) chunk;
    #endif

    // Determine major order of A
    if (A_is_row_major)
    {
        // For row-major (C ordering) matrix A (symmetric or non-symmetric)
        #if defined(USE_OPENMP) && (USE_OPENMP == 1)
        #pragma omp parallel for \
            schedule(static) \
            if (!omp_in_parallel()) \
            default(none) \
            shared(A, b, c, jump, alpha, num_rows, num_columns, chunk, \
                   num_columns_chunked) \
            private(j, sum)
        #endif
        for (long int i=0; i < num_rows; ++i)
        {
            sum = 0.0;
            jump = i * num_columns;
            #if defined(USE_LOOP_UNROLLING) && (USE_LOOP_UNROLLING == 1)
            for (j=0; j < num_columns_chunked; j+= chunk)
            {
                sum += A[jump + j] * b[j] +
                       A[jump + j+1] * b[j+1] +
                       A[jump + j+2] * b[j+2] +
                       A[jump + j+3] * b[j+3] +
                       A[jump + j+4] * b[j+4];
            }
            #endif

            #if defined(USE_LOOP_UNROLLING) && (USE_LOOP_UNROLLING == 1)
            for (j=num_columns_chunked; j < num_columns; ++j)
            #else
            for (j=0; j < num_columns; ++j)
            #endif
            {
                sum += A[jump + j] * b[j];
            }

            c[i] += alpha * static_cast<DataType>(sum);
        }
    }
    else if (A_is_symmetric)
    {
        // A is column-major, but symmetric, so we can use its transposed
        // operation instead, which is more efficient for column-major
        // matrices.
        cMatrixOperations<DataType>::dense_transposed_matvec_plus(
            A, b, alpha, num_rows, num_columns, A_is_row_major, A_is_symmetric,
            c);
    }
    else
    {
        // For column-major (Fortran ordering) non-symmetric matrix A
        #if defined(USE_OPENMP) && (USE_OPENMP == 1)
        #pragma omp parallel for \
            schedule(static) \
            if (!omp_in_parallel()) \
            default(none) \
            shared(A, b, c, alpha, num_rows, num_columns) \
            private(j, sum)
        #endif
        for (long int i=0; i < num_rows; ++i)
        {
            sum = 0.0;
            for (j=0; j < num_columns; ++j)
            {
                sum += A[i + num_rows*j] * b[j];
            }
            c[i] += alpha* static_cast<DataType>(sum);
        }
    }

    #endif
}


// =======================
// dense transposed matvec
// =======================

/// \brief      Computes matrix vector multiplication \f$\boldsymbol{c} =
///             \mathbf{A}^{\intercal} \boldsymbol{b} \f$ where \f$ \mathbf{A}
///             \f$ is dense, and \f$ \mathbf{A}^{\intercal} \f$ is the
///             transpose of the matrix \f$ \mathbf{A} \f$.
///
/// \details    The reduction variable (here, \c sum ) is of the type
///             <tt>long double</tt>. This is becase when \c DataType is \c
///             float, the summation loses the precision, especially when the
///             vector size is large. It seems that using <tt>long double</tt>
///             is slightly faster than using \c double. The advantage of using
///             a type with larger bits for the reduction variable is only
///             sensible if the compiler is optimized with \c -O2 or \c -O3
///             flags.
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
/// \param[in]  A_is_symmetric
///             Boolean. If \c A is symmetric, set this value to \c 1,
///             otherwise \c 0.
/// \param[out] c
///             The output column vector (written in-place).

template <typename DataType>
void cMatrixOperations<DataType>::dense_transposed_matvec(
        const DataType* RESTRICT A,
        const DataType* RESTRICT b,
        const LongIndexType num_rows,
        const LongIndexType num_columns,
        const FlagType A_is_row_major,
        const FlagType A_is_symmetric,
        DataType* RESTRICT c)
{
    #if defined(USE_ANY_CBLAS) && (USE_ANY_CBLAS == 1)

    // Using BLAS
    CBLAS_LAYOUT layout;
    CBLAS_UPLO uplo;
    CBLAS_TRANSPOSE transpose = CblasTrans;
    int lda;
    if (A_is_row_major)
    {
        layout = CblasRowMajor;
        uplo = CblasUpper;
        lda = num_columns;
    }
    else
    {
        layout = CblasColMajor;
        uplo = CblasLower;
        lda = num_rows;

        // For efficiency, use transpose op for symmetric column-major matrices
        if (A_is_symmetric)
        {
            transpose = CblasNoTrans;
        }
    }

    int incb = 1;
    int incc = 1;
    DataType alpha = 1.0;
    DataType beta = 0.0;

    if (A_is_symmetric)
    {
        cblas_api::xsymv(layout, uplo, num_columns, alpha, A, lda, b, incb,
                         beta, c, incc);
    }
    else
    {
        cblas_api::xgemv(layout, transpose, num_rows, num_columns, alpha, A,
                         lda, b, incb, beta, c, incc);
    }

    #else

    // Not using BLAS
    long int i;
    long int jump;
    long double sum;
    const long int chunk = 5;
    const long int num_rows_chunked = num_rows - (num_rows % chunk);

    // Void unused variables to avoid compiler warnings (-Wno-unused-parameter)
    #if !defined(USE_LOOP_UNROLLING) || (USE_LOOP_UNROLLING != 1)
    (void) num_rows_chunked;
    (void) chunk;
    #endif

    // Determine major order of A
    if (!A_is_row_major)
    {
        // For column-major (Fortran ordering) matrix A (symmetric or
        // non-symmetric)
        #if defined(USE_OPENMP) && (USE_OPENMP == 1)
        #pragma omp parallel for \
            schedule(static) \
            if (!omp_in_parallel()) \
            default(none) \
            shared(A, b, c, jump, num_rows, num_columns, num_rows_chunked, \
                   chunk) \
            private(i, sum)
        #endif
        for (long int j=0; j < num_columns; ++j)
        {
            // Loop unrolling
            sum = 0.0;
            jump = num_rows * j;
            #if defined(USE_LOOP_UNROLLING) && (USE_LOOP_UNROLLING == 1)
            for (i=0; i < num_rows_chunked; i += chunk)
            {
                sum += A[i + jump] * b[i] +
                       A[i+1 + jump] * b[i+1] +
                       A[i+2 + jump] * b[i+2] +
                       A[i+3 + jump] * b[i+3] +
                       A[i+4 + jump] * b[i+4];
            }
            #endif

            #if defined(USE_LOOP_UNROLLING) && (USE_LOOP_UNROLLING == 1)
            for (i=num_rows_chunked; i < num_rows; ++i)
            #else
            for (i=0; i < num_rows; ++i)
            #endif
            {
                sum += A[i + jump] * b[i];
            }

            c[j] = static_cast<DataType>(sum);
        }
    }
    else if (A_is_symmetric)
    {
        // A is row-major, but symmetric, so we can use its non-transposed
        // operation instead, which is more efficient for row-major
        // matrices.
        cMatrixOperations<DataType>::dense_matvec(
            A, b, num_rows, num_columns, A_is_row_major, A_is_symmetric, c);
    }
    else
    {
        // For row-major (C ordering) non-symmetric matrix A
        #if defined(USE_OPENMP) && (USE_OPENMP == 1)
        #pragma omp parallel for \
            schedule(static) \
            if (!omp_in_parallel()) \
            default(none) \
            shared(A, b, c, num_rows, num_columns) \
            private(i, sum)
        #endif
        for (long int j=0; j < num_columns; ++j)
        {
            sum = 0.0;
            for (i=0; i < num_rows; ++i)
            {
                sum += A[i*num_columns + j] * b[i];
            }
            c[j] = static_cast<DataType>(sum);
        }
    }

    #endif
}


// ============================
// dense transposed matvec plus
// ============================

/// \brief         Computes \f$ \boldsymbol{c} = \boldsymbol{c} + \alpha
///                \mathbf{A}^{\intercal} \boldsymbol{b} \f$ where \f$
///                \mathbf{A} \f$ is dense, and \f$ \mathbf{A}^{\intercal} \f$
///                is the transpose of the matrix \f$ \mathbf{A} \f$.
///
/// \details       The reduction variable (here, \c sum ) is of the type
///                <tt>long double</tt>. This is becase when \c DataType is \c
///                float the summation loses the precision, especially when the
///                vector size is large. It seems that using <tt>long double
///                </tt> is slightly faster than using \c double. The advantage
///                of using a type with larger bits for the reduction variable
///                is only sensible if the compiler is optimized with \c -O2 or
///                \c -O3 flags.
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
/// \param[in]     A_is_symmetric
///                Boolean. If \c A is symmetric, set this value to \c 1,
///                otherwise \c 0.
/// \param[in,out] c
///                The output column vector (written in-place).

template <typename DataType>
void cMatrixOperations<DataType>::dense_transposed_matvec_plus(
        const DataType* RESTRICT A,
        const DataType* RESTRICT b,
        const DataType alpha,
        const LongIndexType num_rows,
        const LongIndexType num_columns,
        const FlagType A_is_row_major,
        const FlagType A_is_symmetric,
        DataType* RESTRICT c)
{
    #if defined(USE_ANY_CBLAS) && (USE_ANY_CBLAS == 1)

    // Using BLAS
    CBLAS_LAYOUT layout;
    CBLAS_TRANSPOSE transpose = CblasTrans;
    CBLAS_UPLO uplo;
    int lda;
    if (A_is_row_major)
    {
        layout = CblasRowMajor;
        uplo = CblasUpper;
        lda = num_columns;
    }
    else
    {
        layout = CblasColMajor;
        uplo = CblasLower;
        lda = num_rows;

        // For efficiency, use transpose op for symmetric column-major matrices
        if (A_is_symmetric)
        {
            transpose = CblasNoTrans;
        }
    }

    int incb = 1;
    int incc = 1;
    DataType beta = 1.0;

    if (A_is_symmetric)
    {
        cblas_api::xsymv(layout, uplo, num_columns, alpha, A, lda, b, incb,
                         beta, c, incc);
    }
    else
    {
        cblas_api::xgemv(layout, transpose, num_rows, num_columns, alpha, A,
                         lda, b, incb, beta, c, incc);
    }

    #else

    // Not using BLAS
    DataType zero = 0.0;
    if (c_arithmetics::is_equal(alpha, zero))
    {
        return;
    }

    long int i;
    long int jump;
    long double sum;
    const long int chunk = 5;
    const long int num_rows_chunked = num_rows - (num_rows % chunk);

    // Void unused variables to avoid compiler warnings (-Wno-unused-parameter)
    #if !defined(USE_LOOP_UNROLLING) || (USE_LOOP_UNROLLING != 1)
    (void) num_rows_chunked;
    (void) chunk;
    #endif

    // Determine major order of A
    if (!A_is_row_major)
    {
        // For column-major (Fortran ordering) matrix A (symmetric or
        // non-symmetric)
        #if defined(USE_OPENMP) && (USE_OPENMP == 1)
        #pragma omp parallel for \
            schedule(static) \
            if (!omp_in_parallel()) \
            default(none) \
            shared(A, b, c, jump, alpha, num_rows, num_columns, \
                   num_rows_chunked, chunk) \
            private(i, sum)
        #endif
        for (long int j=0; j < num_columns; ++j)
        {
            sum = 0.0;
            jump = num_rows * j;
            #if defined(USE_LOOP_UNROLLING) && (USE_LOOP_UNROLLING == 1)
            for (i=0; i < num_rows_chunked; i += chunk)
            {
                sum += A[i + jump] * b[i] +
                       A[i+1 + jump] * b[i+1] +
                       A[i+2 + jump] * b[i+2] +
                       A[i+3 + jump] * b[i+3] +
                       A[i+4 + jump] * b[i+4];
            }
            #endif

            #if defined(USE_LOOP_UNROLLING) && (USE_LOOP_UNROLLING == 1)
            for (i=num_rows_chunked; i < num_rows; ++i)
            #else
            for (i=0; i < num_rows; ++i)
            #endif
            {
                sum += A[i + jump] * b[i];
            }

            c[j] += alpha * static_cast<DataType>(sum);
        }
    }
    else if (A_is_symmetric)
    {
        // A is row-major, but symmetric, so we can use its non-transposed
        // operation instead, which is more efficient for row-major
        // matrices.
        cMatrixOperations<DataType>::dense_matvec_plus(
            A, b, alpha, num_rows, num_columns, A_is_row_major, A_is_symmetric,
            c);
    }
    else
    {
        // For row-major (C ordering) non-symmetric matrix A
        #if defined(USE_OPENMP) && (USE_OPENMP == 1)
        #pragma omp parallel for \
            schedule(static) \
            if (!omp_in_parallel()) \
            default(none) \
            shared(A, b, c, alpha, num_rows, num_columns) \
            private(i, sum)
        #endif
        for (long int j=0; j < num_columns; ++j)
        {
            sum = 0.0;
            for (i=0; i < num_rows; ++i)
            {
                sum += A[i*num_columns + j] * b[i];
            }
            c[j] += alpha * static_cast<DataType>(sum);
        }
    }

    #endif
}


// ==========
// csr matvec
// ==========

/// \brief      Computes \f$ \boldsymbol{c} = \mathbf{A} \boldsymbol{b} \f$
///             where \f$ \mathbf{A} \f$ is compressed sparse row (CSR) matrix
///             and \f$ \boldsymbol{b} \f$ is a dense vector. The output \f$
///             \boldsymbol{c} \f$ is a dense vector.
///
/// \details    The reduction variable (here, \c sum ) is of the type
///             <tt>long double</tt>. This is becase when \c DataType is \c
///             float, the summation loses the precision, especially when the
///             vector size is large. It seems that using <tt>long double</tt>
///             is slightly faster than using \c double. The advantage of using
///             a type with larger bits for the reduction variable is only
///             sensible if the compiler is optimized with \c -O2 or \c -O3
///             flags.
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
void cMatrixOperations<DataType>::csr_matvec(
        const DataType* RESTRICT A_data,
        const LongIndexType* RESTRICT A_column_indices,
        const LongIndexType* RESTRICT A_index_pointer,
        const DataType* RESTRICT b,
        const LongIndexType num_rows,
        DataType* RESTRICT c)
{
    LongIndexType index_pointer;
    LongIndexType row;
    LongIndexType column;
    long double sum;

    #if defined(USE_OPENMP) && (USE_OPENMP == 1)
    #pragma omp parallel for \
        schedule(static) \
        if (!omp_in_parallel()) \
        default(none) \
        shared(A_data, A_column_indices, A_index_pointer, b, c, num_rows) \
        private(index_pointer, column, sum)
    #endif
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
        c[row] = static_cast<DataType>(sum);
    }
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
/// \details       The reduction variable (here, \c sum ) is of the type
///                <tt>long double</tt>. This is becase when \c DataType is \c
///                float the summation loses the precision, especially when the
///                vector size is large. It seems that using <tt>long double
///                </tt> is slightly faster than using \c double. The advantage
///                of using a type with larger bits for the reduction variable
///                is only sensible if the compiler is optimized with \c -O2 or
///                \c -O3 flags.
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
void cMatrixOperations<DataType>::csr_matvec_plus(
        const DataType* A_data,
        const LongIndexType* RESTRICT A_column_indices,
        const LongIndexType* RESTRICT A_index_pointer,
        const DataType* RESTRICT b,
        const DataType alpha,
        const LongIndexType num_rows,
        DataType* RESTRICT c)
{
    DataType zero = 0.0;
    if (c_arithmetics::is_equal(alpha, zero))
    {
        return;
    }

    LongIndexType index_pointer;
    LongIndexType row;
    LongIndexType column;
    long double sum;

    #if defined(USE_OPENMP) && (USE_OPENMP == 1)
    #pragma omp parallel for \
        schedule(static) \
        if (!omp_in_parallel()) \
        default(none) \
        shared(A_data, A_column_indices, A_index_pointer, b, c, alpha, \
               num_rows) \
        private(index_pointer, column, sum)
    #endif
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
        c[row] += alpha * static_cast<DataType>(sum);
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
/// \param[in]  A_is_symmetric
///             Boolean. If \c A is symmetric, set this value to \c 1,
///             otherwise \c 0.
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
void cMatrixOperations<DataType>::csr_transposed_matvec(
        const DataType* RESTRICT A_data,
        const LongIndexType* RESTRICT A_column_indices,
        const LongIndexType* RESTRICT A_index_pointer,
        const FlagType A_is_symmetric,
        const DataType* RESTRICT b,
        const LongIndexType num_rows,
        const LongIndexType num_columns,
        DataType* RESTRICT c)
{
    if (A_is_symmetric)
    {
        // For symmetric A, use non-transposed operation instead for efficiency
        cMatrixOperations<DataType>::csr_matvec(
            A_data, A_column_indices, A_index_pointer, b, num_rows, c);
    }
    else
    {
        // A is non-symmetric, use transposed product operation
        LongIndexType index_pointer;
        LongIndexType row;
        LongIndexType column;

        // Initialize output to zero
        #if defined(USE_OPENMP) && (USE_OPENMP == 1)
        #pragma omp parallel for \
            schedule(static) \
            if ((!omp_in_parallel()) && (num_columns >= LARGE_ARRAY_SIZE)) \
            default(none) shared(c, num_columns)
        #endif
        for (column=0; column < num_columns; ++column)
        {
            c[column] = 0.0;
        }

        #if defined(USE_OPENMP) && (USE_OPENMP == 1)
        #pragma omp parallel for \
            schedule(static) \
            if (!omp_in_parallel()) \
            default(none) \
            shared(A_data, A_column_indices, A_index_pointer, b, c, num_rows) \
            private(index_pointer, column)
        #endif
        for (row=0; row < num_rows; ++row)
        {
            for (index_pointer=A_index_pointer[row];
                 index_pointer < A_index_pointer[row+1];
                 ++index_pointer)
            {
                column = A_column_indices[index_pointer];

                #if defined(USE_OPENMP) && (USE_OPENMP == 1)
                #pragma omp atomic update
                #endif
                c[column] += A_data[index_pointer] * b[row];
            }
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
/// \param[in]     A_is_symmetric
///                Boolean. If \c A is symmetric, set this value to \c 1,
///                otherwise \c 0.
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
void cMatrixOperations<DataType>::csr_transposed_matvec_plus(
        const DataType* RESTRICT A_data,
        const LongIndexType* RESTRICT A_column_indices,
        const LongIndexType* RESTRICT A_index_pointer,
        const FlagType A_is_symmetric,
        const DataType* RESTRICT b,
        const DataType alpha,
        const LongIndexType num_rows,
        DataType* RESTRICT c)
{
    DataType zero = 0.0;
    if (c_arithmetics::is_equal(alpha, zero))
    {
        return;
    }

    if (A_is_symmetric)
    {
        // For symmetric A, use non-transposed operation instead for efficiency
        cMatrixOperations<DataType>::csr_matvec_plus(
            A_data, A_column_indices, A_index_pointer, b, alpha, num_rows, c);
    }
    else
    {
        // A is non-symmetric, use transposed product operation
        LongIndexType index_pointer;
        LongIndexType row;
        LongIndexType column;

        #if defined(USE_OPENMP) && (USE_OPENMP == 1)
        #pragma omp parallel for \
            schedule(static) \
            if (!omp_in_parallel()) \
            default(none) \
            shared(A_data, A_column_indices, A_index_pointer, b, c, alpha, \
                   num_rows) \
            private(index_pointer, column)
        #endif
        for (row=0; row < num_rows; ++row)
        {
            for (index_pointer=A_index_pointer[row];
                 index_pointer < A_index_pointer[row+1];
                 ++index_pointer)
            {
                column = A_column_indices[index_pointer];

                #if defined(USE_OPENMP) && (USE_OPENMP == 1)
                #pragma omp atomic update
                #endif
                c[column] += alpha * A_data[index_pointer] * b[row];
            }
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
/// \param[in]  A_is_symmetric
///             Boolean. If \c A is symmetric, set this value to \c 1,
///             otherwise \c 0.
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
void cMatrixOperations<DataType>::csc_matvec(
        const DataType* RESTRICT A_data,
        const LongIndexType* RESTRICT A_row_indices,
        const LongIndexType* RESTRICT A_index_pointer,
        const FlagType A_is_symmetric,
        const DataType* RESTRICT b,
        const LongIndexType num_rows,
        const LongIndexType num_columns,
        DataType* RESTRICT c)
{
    if (A_is_symmetric)
    {
        // For symmetric A, use transposed operation instead for efficiency
        cMatrixOperations<DataType>::csc_transposed_matvec(
            A_data, A_row_indices, A_index_pointer, b, num_columns, c);
    }
    else
    {
        // A is non-symmetric, use non-transposed product operation
        LongIndexType index_pointer;
        LongIndexType row;
        LongIndexType column;

        // Initialize output to zero
        #if defined(USE_OPENMP) && (USE_OPENMP == 1)
        #pragma omp parallel for \
            schedule(static) \
            if (!omp_in_parallel() && (num_rows >= LARGE_ARRAY_SIZE)) \
            default(none) \
            shared(c, num_rows)
        #endif
        for (row=0; row < num_rows; ++row)
        {
            c[row] = 0.0;
        }

        #if defined(USE_OPENMP) && (USE_OPENMP == 1)
        #pragma omp parallel for \
            schedule(static) \
            if (!omp_in_parallel()) \
            default(none) \
            shared(A_data, A_row_indices, A_index_pointer, b, c, num_columns) \
            private(index_pointer, row)
        #endif
        for (column=0; column < num_columns; ++column)
        {
            for (index_pointer=A_index_pointer[column];
                 index_pointer < A_index_pointer[column+1];
                 ++index_pointer)
            {
                row = A_row_indices[index_pointer];

                #if defined(USE_OPENMP) && (USE_OPENMP == 1)
                #pragma omp atomic update
                #endif
                c[row] += A_data[index_pointer] * b[column];
            }
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
/// \param[in]     A_is_symmetric
///                Boolean. If \c A is symmetric, set this value to \c 1,
///                otherwise \c 0.
/// \param[in]     b
///                Column vector with same size of the number of columns of
///                \c A.
/// \param[in]     alpha
///                A scalar that scales the matrix vector multiplication.
/// \param[in]     num_columns
///                Number of columns of the matrix \c A. This is essentially
///                the size of \c A_index_pointer array minus one.
/// \param[in,out] c
///                Output column vector with the same size as \c b. This array
///                is written in-place.

template <typename DataType>
void cMatrixOperations<DataType>::csc_matvec_plus(
        const DataType* RESTRICT A_data,
        const LongIndexType* RESTRICT A_row_indices,
        const LongIndexType* RESTRICT A_index_pointer,
        const FlagType A_is_symmetric,
        const DataType* RESTRICT b,
        const DataType alpha,
        const LongIndexType num_columns,
        DataType* RESTRICT c)
{
    DataType zero = 0.0;
    if (c_arithmetics::is_equal(alpha, zero))
    {
        return;
    }

    if (A_is_symmetric)
    {
        // For symmetric A, use transposed operation instead for efficiency
        cMatrixOperations<DataType>::csc_transposed_matvec_plus(
            A_data, A_row_indices, A_index_pointer, b, alpha, num_columns, c);
        
    }
    else
    {
        LongIndexType index_pointer;
        LongIndexType row;
        LongIndexType column;

        #if defined(USE_OPENMP) && (USE_OPENMP == 1)
        #pragma omp parallel for \
            schedule(static) \
            if (!omp_in_parallel()) \
            default(none) \
            shared(A_data, A_row_indices, A_index_pointer, b, c, alpha, \
                   num_columns) \
            private(index_pointer, row)
        #endif
        for (column=0; column < num_columns; ++column)
        {
            for (index_pointer=A_index_pointer[column];
                 index_pointer < A_index_pointer[column+1];
                 ++index_pointer)
            {
                row = A_row_indices[index_pointer];

                #if defined(USE_OPENMP) && (USE_OPENMP == 1)
                #pragma omp atomic update
                #endif
                c[row] += alpha * A_data[index_pointer] * b[column];
            }
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
/// \details    The reduction variable (here, \c sum ) is of the type
///             <tt>long double</tt>. This is becase when \c DataType is \c
///             float, the summation loses the precision, especially when the
///             vector size is large. It seems that using <tt>long double</tt>
///             is slightly faster than using \c double. The advantage of using
///             a type with larger bits for the reduction variable is only
///             sensible if the compiler is optimized with \c -O2 or \c -O3
///             flags.
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
void cMatrixOperations<DataType>::csc_transposed_matvec(
        const DataType* RESTRICT A_data,
        const LongIndexType* RESTRICT A_row_indices,
        const LongIndexType* RESTRICT A_index_pointer,
        const DataType* RESTRICT b,
        const LongIndexType num_columns,
        DataType* RESTRICT c)
{
    LongIndexType index_pointer;
    LongIndexType row;
    LongIndexType column;
    long double sum;

    #if defined(USE_OPENMP) && (USE_OPENMP == 1)
    #pragma omp parallel for \
        schedule(static) \
        if (!omp_in_parallel()) \
        default(none) \
        shared(A_data, A_row_indices, A_index_pointer, b, c, num_columns) \
        private(index_pointer, row, sum)
    #endif
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
        c[column] = static_cast<DataType>(sum);
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
/// \details       The reduction variable (here, \c sum ) is of the type
///                <tt>long double</tt>. This is becase when \c DataType is \c
///                float the summation loses the precision, especially when the
///                vector size is large. It seems that using <tt>long double
///                </tt> is slightly faster than using \c double. The advantage
///                of using a type with larger bits for the reduction variable
///                is only sensible if the compiler is optimized with \c -O2 or
///                \c -O3 flags.
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
void cMatrixOperations<DataType>::csc_transposed_matvec_plus(
        const DataType* RESTRICT A_data,
        const LongIndexType* RESTRICT A_row_indices,
        const LongIndexType* RESTRICT A_index_pointer,
        const DataType* RESTRICT b,
        const DataType alpha,
        const LongIndexType num_columns,
        DataType* RESTRICT c)
{
    DataType zero = 0.0;
    if (c_arithmetics::is_equal(alpha, zero))
    {
        return;
    }

    LongIndexType index_pointer;
    LongIndexType row;
    LongIndexType column;
    long double sum;

    #if defined(USE_OPENMP) && (USE_OPENMP == 1)
    #pragma omp parallel for \
        schedule(static) \
        if (!omp_in_parallel()) \
        default(none) \
        shared(A_data, A_row_indices, A_index_pointer, b, c, alpha, \
               num_columns) \
        private(index_pointer, row, sum)
    #endif
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
        c[column] += static_cast<DataType>(alpha * sum);
    }
}


// ==================
// create band matrix
// ==================

/// \brief      Creates bi-diagonal or symmetric tri-diagonal matrix from the
///             diagonal array (\c diagonals) and off-diagonal array (\c
///             supdiagonals).
///
/// \details    The output is written in place (in \c A). The output is only
///             written up to the \c non_zero_size element, that is: \c
///             A[:non_zero_size,:non_zero_size] is filled, and the rest
///             is assumed to be zero.
///
///             Depending on \c tridiagonal, the matrix is upper bi-diagonal or
///             symmetric tri-diagonal.
///
/// \note       The matrix \c A is assumed to be initialized to zero before
///             calling this function.
///
/// \param[out] A
///             1D array that represents a 2D dense array with either C (row)
///             major ordering or Fortran (column) major ordering. The major
///             ordering should de defined by \c A_is_row_major flag.
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
/// \param[in]  diagonals
///             An array of length \c n. All elements \c diagonals create the
///             diagonals of \c A.
/// \param[in]  supdiagonals
///             An array of length \c n. Elements \c supdiagonals[0:-1] create
///             the upper off-diagonal of \c A, making \c A an upper
///             bi-diagonal matrix. In addition, if \c tridiagonal is set to
///             \c 1, the lower off-diagonal is also created similar to the
///             upper off-diagonal, making \c A a symmetric tri-diagonal
///             matrix.
/// \param[in]  non_zero_size
///             Up to the \c A[:non_zero_size,:non_zero_size] of \c A will be
///             written. At most, \c non_zero_size can be \c n, which is the
///             size of \c diagonals array and the size of the square matrix.
///             If \c non_zero_size is less than \c n, it is due to the fact
///             that either \c diagonals or \c supdiagonals has zero elements
///             after the \c size element (possibly due to early termination
///             of Lanczos iterations method).
/// \param[in]  tridiagonal
///             Boolean. If set to \c 0, the matrix \c T becomes upper
///             bi-diagonal. If set to \c 1, the matrix becomes symmetric
///             tri-diagonal.

template <typename DataType>
void cMatrixOperations<DataType>::create_band_matrix(
        DataType* RESTRICT A,
        const LongIndexType num_rows,
        const LongIndexType num_columns,
        const FlagType A_is_row_major,
        const DataType* RESTRICT diagonals,
        const DataType* RESTRICT supdiagonals,
        const IndexType non_zero_size,
        const FlagType tridiagonal)
{
    if (A_is_row_major)
    {
        // A is row-major
        #if defined(USE_OPENMP) && (USE_OPENMP == 1)
        #pragma omp parallel for \
            schedule(static) \
            if (!omp_in_parallel() && (non_zero_size >= LARGE_ARRAY_SIZE)) \
            default(none) \
            shared(A, diagonals, supdiagonals, non_zero_size, tridiagonal, \
                   num_columns)
        #endif
        for (IndexType j=0; j < non_zero_size; ++j)
        {
            // Diagonals
            A[j*num_columns + j] = diagonals[j];

            // Off diagonals
            if (j < non_zero_size-1)
            {
                // Sup-diagonal
                A[j*num_columns + j+1] = supdiagonals[j];

                // Sub-diagonal, making symmetric tri-diagonal matrix
                if (tridiagonal)
                {
                    A[(j+1)*num_columns + j] = supdiagonals[j];
                }
            }
        }
    }
    else
    {
        // A is column-major
        #if defined(USE_OPENMP) && (USE_OPENMP == 1)
        #pragma omp parallel for \
            schedule(static) \
            if (!omp_in_parallel() && (non_zero_size >= LARGE_ARRAY_SIZE)) \
            default(none) \
            shared(A, diagonals, supdiagonals, non_zero_size, tridiagonal, \
                   num_rows)
        #endif
        for (IndexType j=0; j < non_zero_size; ++j)
        {
            // Diagonals
            A[j + num_rows*j] = diagonals[j];

            // Off diagonals
            if (j < non_zero_size-1)
            {
                // Sup-diagonal
                A[j + (j+1)*num_rows] = supdiagonals[j];

                // Sub-diagonal, making symmetric tri-diagonal matrix
                if (tridiagonal)
                {
                    A[j+1 + j*num_rows] = supdiagonals[j];
                }
            }
        }
    }
}


// ===============================
// Explicit template instantiation
// ===============================

template class cMatrixOperations<float>;
template class cMatrixOperations<double>;
template class cMatrixOperations<long double>;
