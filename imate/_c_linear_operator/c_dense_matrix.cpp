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

#include "./c_dense_matrix.h"
#include <cstddef>  // NULL
#include "../_definitions/definitions.h"  // USE_OPENMP
#if defined(USE_OPENMP) && (USE_OPENMP == 1)
    #include <omp.h>  // omp_in_parallel
#endif
#include "../_c_arithmetics/c_arithmetics.h"  // c_arithmetics
#include "../_c_basic_algebra/c_matrix_operations.h"  // cMatrixOperations


// =============
// constructor 1
// =============

/// \brief Default constructor.
///

template <typename DataType>
cDenseMatrix<DataType>::cDenseMatrix():
    
    // Initializer list
    A(NULL),
    A_is_row_major(0)
{
}


// =============
// constructor 2
// =============

/// \brief      Constructor.
///
/// \param[in]  A_
///             1D array that represents a 2D dense array with either C (row)
///             major ordering or Fortran (column) major ordering. The major
///             ordering should de defined by \c A_is_row_major flag.
/// \param[in]  num_rows_
///             Number of rows of \c A
/// \param[in]  num_columns_
///             Number of columns of \c A
/// \param[in]  A_is_row_major_
///             Boolean, can be \c 0 or \c 1 as follows:
///             * If \c A is row major (C ordering where the last index is
///               contiguous) this value should be \c 1.
///             * If \c A is column major (Fortran ordering where the first
///               index is contiguous), this value should be set to \c 0.
/// \param[in]  A_is_symmetric_
///             Boolean. If \c A is symmetric, set this value to \c 1,
///             otherwise \c 0.

template <typename DataType>
cDenseMatrix<DataType>::cDenseMatrix(
        const DataType* A_,
        const LongIndexType num_rows_,
        const LongIndexType num_columns_,
        const FlagType A_is_row_major_,
        const FlagType A_is_symmetric_):

    // Base class constructors
    cLinearOperatorBase(num_rows_, num_columns_),
    cMatrix<DataType>(A_is_symmetric_),

    // Initializer list
    A(A_),
    A_is_row_major(A_is_row_major_)
{
}


// ==========
// destructor
// ==========

/// \brief Destructor.
///

template <typename DataType>
cDenseMatrix<DataType>::~cDenseMatrix()
{
}


// ==================
// is identity matrix
// ==================

/// \brief   Checks whether the matrix is identity.
///
/// \details The identity check is primarily performed in the \c
///          cAffineMatrixFunction class.
///
/// \return  Returns \c 1 if the input matrix is identity, and \c 0 otherwise.
///
/// \sa      cAffineMatrixFunction

template <typename DataType>
FlagType cDenseMatrix<DataType>::is_identity_matrix() const
{
    FlagType matrix_is_identity = 1;
    DataType matrix_element;
    const DataType diagonal = 1.0;
    const DataType off_diagonal = 0.0;

    // Check matrix element-wise
    if (this->A_is_row_major)
    {
        // Row-major matrix
        LongIndexType column;
        LongIndexType num_checking_columns;

        #if defined(USE_OPENMP) && (USE_OPENMP == 1)
        #pragma omp parallel for \
            schedule(static) \
            if (!omp_in_parallel()) \
            default(none) \
            shared(matrix_is_identity, diagonal, off_diagonal) \
            private(column, num_checking_columns, matrix_element)
        #endif
        for (LongIndexType row=0; row < this->num_rows; ++row)
        {
            if (matrix_is_identity)
            {
                if (this->A_is_symmetric)
                {
                    // Check only half of the columns up to diagonal element
                    num_checking_columns = row + 1;
                }
                else
                {
                    num_checking_columns = this->num_columns;
                }

                for (column=0; column < num_checking_columns; ++column)
                {
                    // Get an element of the matrix
                    matrix_element = this->A[row * this->num_columns + column];

                    // Check the value of element with identity matrix
                    if (((row == column) && \
                         (!c_arithmetics::is_equal(matrix_element,
                                                   diagonal))) || \
                        ((row != column) && \
                         (!c_arithmetics::is_equal(matrix_element,
                                                   off_diagonal))))
                    {
                        #if defined(USE_OPENMP) && (USE_OPENMP == 1)
                        #pragma omp atomic write
                        #endif
                        matrix_is_identity = 0;

                        break;
                    }
                }
            }
        }
    }
    else
    {
        // Column-major matrix
        LongIndexType row;
        LongIndexType num_checking_rows;

        #if defined(USE_OPENMP) && (USE_OPENMP == 1)
        #pragma omp parallel for \
            schedule(static) \
            if (!omp_in_parallel()) \
            default(none) \
            shared(matrix_is_identity, diagonal, off_diagonal) \
            private(row, num_checking_rows, matrix_element)
        #endif
        for (LongIndexType column=0; column < this-> num_columns; ++column)
        {
            if (matrix_is_identity)
            {
                if (this->A_is_symmetric)
                {
                    // Check only half of the rows up to diagonal element
                    num_checking_rows = column + 1;
                }
                else
                {
                    num_checking_rows = this->num_rows;
                }

                for (row=0; row < num_checking_rows; ++row)
                {
                    // Get an element of the matrix
                    matrix_element = this->A[column * this->num_rows + row];

                    // Check the value of element with identity matrix
                    if (((row == column) && \
                         (!c_arithmetics::is_equal(matrix_element,
                                                   diagonal))) || \
                        ((row != column) && \
                         (!c_arithmetics::is_equal(matrix_element,
                                                   off_diagonal))))
                    {
                        #if defined(USE_OPENMP) && (USE_OPENMP == 1)
                        #pragma omp atomic write
                        #endif
                        matrix_is_identity = 0;

                        break;
                    }
                }
            }
        }
    }

    return matrix_is_identity;
}


// ===
// dot
// ===

/// \brief      Matrix vector product.
///
/// \details    Performs the matrix vector product \f$ \boldsymbol{y} =
///             \mathbf{A} \boldsymbol{x} \f$.
///
/// \param[in]  vector
///             A one-dimensional input vector \f$ \boldsymbol{x} \f$ with size
///             the of the number of columns of the matrix \f$ \mathbf{A} \f$.
/// \param[out] product
///             A one-dimensional output vector \f$ \boldsymbol{y} \f$ with the
///             size of the number of rows of \f$ \mathbf{A} \f$. This vector
///             will be overwritten.
///
/// \sa         cDenseMatrix::dot_plus,
///             cDenseMatrix::transposed_dot
///             cDenseMatrix::transposed_dot_plus

template <typename DataType>
void cDenseMatrix<DataType>::dot(
        const DataType* vector,
        DataType* product)
{
    cMatrixOperations<DataType>::dense_matvec(
            this->A,
            vector,
            this->num_rows,
            this->num_columns,
            this->A_is_row_major,
            this->A_is_symmetric,
            product);
}


// ========
// dot plus
// ========

/// \brief      Matrix vector product written in place.
///
/// \details    Performs the matrix vector product \f$ \boldsymbol{y} =
///             \boldsymbol{y} + \alpha \mathbf{A} \boldsymbol{x} \f$.
///
/// \param[in]  vector
///             A one-dimensional input vector \f$ \boldsymbol{x} \f$ with size
///             the of the number of columns of the matrix \f$ \mathbf{A} \f$.
/// \param[in]  alpha
///             A scalar.
/// \param[out] product
///             A one-dimensional output vector \f$ \boldsymbol{y} \f$ with the
///             size of the number of rows of \f$ \mathbf{A} \f$.
///
/// \sa         cDenseMatrix::dot,
///             cDenseMatrix::transposed_dot
///             cDenseMatrix::transposed_dot_plus

template <typename DataType>
void cDenseMatrix<DataType>::dot_plus(
        const DataType* vector,
        const DataType alpha,
        DataType* product)
{
    cMatrixOperations<DataType>::dense_matvec_plus(
            this->A,
            vector,
            alpha,
            this->num_rows,
            this->num_columns,
            this->A_is_row_major,
            this->A_is_symmetric,
            product);
}


// =============
// transpose dot
// =============

/// \brief      Transposed-matrix vector product.
///
/// \details    Performs the matrix vector product:
///             \f$ \boldsymbol{y} = \mathbf{A}^{\intercal} \boldsymbol{x} \f$.
///
/// \param[in]  vector
///             A one-dimensional input vector \f$ \boldsymbol{x} \f$ with size
///             the of the number of columns of the matrix \f$ \mathbf{A} \f$.
/// \param[out] product
///             A one-dimensional output vector \f$ \boldsymbol{y} \f$ with the
///             size of the number of rows of \f$ \mathbf{A} \f$. This vector
///             will be overwritten.
///
/// \sa         cDenseMatrix::dot_plus,
///             cDenseMatrix::dot
///             cDenseMatrix::transposed_dot_plus

template <typename DataType>
void cDenseMatrix<DataType>::transpose_dot(
        const DataType* vector,
        DataType* product)
{
    cMatrixOperations<DataType>::dense_transposed_matvec(
            this->A,
            vector,
            this->num_rows,
            this->num_columns,
            this->A_is_row_major,
            this->A_is_symmetric,
            product);
}


// ==================
// transpose dot plus
// ==================

/// \brief      Transposed-matrix vector product written in place.
///
/// \details    Performs the matrix vector product \f$ \boldsymbol{y} =
///             \boldsymbol{y} + \alpha \mathbf{A}^{\intercal} \boldsymbol{x}
///             \f$.
///
/// \param[in]  vector
///             A one-dimensional input vector \f$ \boldsymbol{x} \f$ with size
///             the of the number of columns of the matrix \f$ \mathbf{A} \f$.
/// \param[in]  alpha
///             A scalar.
/// \param[out] product
///             A one-dimensional output vector \f$ \boldsymbol{y} \f$ with the
///             size of the number of rows of \f$ \mathbf{A} \f$.
///
/// \sa         cDenseMatrix::dot_plus,
///             cDenseMatrix::transposed_dot
///             cDenseMatrix::dot

template <typename DataType>
void cDenseMatrix<DataType>::transpose_dot_plus(
        const DataType* vector,
        const DataType alpha,
        DataType* product)
{
    cMatrixOperations<DataType>::dense_transposed_matvec_plus(
            this->A,
            vector,
            alpha,
            this->num_rows,
            this->num_columns,
            this->A_is_row_major,
            this->A_is_symmetric,
            product);
}


// ===============================
// Explicit template instantiation
// ===============================

template class cDenseMatrix<float>;
template class cDenseMatrix<double>;
template class cDenseMatrix<long double>;
