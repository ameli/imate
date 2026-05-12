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

#include "./c_csc_matrix.h"
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
cCSCMatrix<DataType>::cCSCMatrix():
    A_data(NULL),
    A_indices(NULL),
    A_index_pointer(NULL)
{
}


// =============
// constructor 2
// =============

/// \brief      Constructor.
///
/// \param[in]  A_data_
///             1D array of the data content of sparse matrix. The size of the
///             array is the nnz of the matrix.
/// \param[in]  A_indices_
///             1D array indicating the row of each element in \c A_data_ .
///             The size of this array is the nnz of the matrix.
/// \param[in]  A_index_pointer_
///             1D array pointing to the start of new rows in \c
///             A_indices_ . The size of this array is \c num_rows+1 .
///             The first element of this array is \c 0 and the last element
///             of this array is the nnz of the matrix.
/// \param[in]  num_rows_
///             Number of rows of \c A
/// \param[in]  num_columns_
///             Number of columns of \c A
/// \param[in]  A_is_symmetric_
///             Boolean. If \c A is symmetric, set this value to \c 1,
///             otherwise \c 0.

template <typename DataType>
cCSCMatrix<DataType>::cCSCMatrix(
        const DataType* A_data_,
        const LongIndexType* A_indices_,
        const LongIndexType* A_index_pointer_,
        const LongIndexType num_rows_,
        const LongIndexType num_columns_,
        const FlagType A_is_symmetric_):

    // Base class constructors
    cLinearOperatorBase(num_rows_, num_columns_),
    cMatrix<DataType>(A_is_symmetric_),

    // Initializer list
    A_data(A_data_),
    A_indices(A_indices_),
    A_index_pointer(A_index_pointer_)
{
}


// ==========
// destructor
// ==========

/// \brief Destructor.
///

template <typename DataType>
cCSCMatrix<DataType>::~cCSCMatrix()
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
FlagType cCSCMatrix<DataType>::is_identity_matrix() const
{
    FlagType matrix_is_identity = 1;
    LongIndexType index_pointer;
    LongIndexType row;
    DataType matrix_element;
    const DataType diagonal = 1.0;
    const DataType off_diagonal = 0.0;

    // Check matrix element-wise
    #if defined(USE_OPENMP) && (USE_OPENMP == 1)
    #pragma omp parallel for \
        schedule(static) \
        if (!omp_in_parallel()) \
        default(none) \
        shared(matrix_is_identity, diagonal, off_diagonal) \
        private(index_pointer, row, matrix_element)
    #endif
    for (LongIndexType column=0; column < this->num_columns; ++column)
    {
        if (matrix_is_identity)
        {
            for (index_pointer=this->A_index_pointer[column];
                 index_pointer < this->A_index_pointer[column+1];
                 ++index_pointer)
            {
                row = this->A_indices[index_pointer];

                if (!((this->A_is_symmetric) && (column >= row)))
                {
                    matrix_element = this->A_data[index_pointer];

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


// =======
// get nnz
// =======

/// \brief  Returns the number of non-zero elements of the sparse matrix.
///
/// \details The nnz of a CSC matrix can be obtained from the last element of
///          \c A_index_pointer. The size of array \c A_index_pointer is one
///          plus the number of columns of the matrix.
///
/// \return  The nnz of the matrix.

template <typename DataType>
LongIndexType cCSCMatrix<DataType>::get_nnz() const
{
    return this->A_index_pointer[this->num_columns];
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
/// \sa         cCSCMatrix::dot_plus,
///             cCSCMatrix::transposed_dot
///             cCSCMatrix::transposed_dot_plus

template <typename DataType>
void cCSCMatrix<DataType>::dot(
        const DataType* vector,
        DataType* product)
{
    cMatrixOperations<DataType>::csc_matvec(
            this->A_data,
            this->A_indices,
            this->A_index_pointer,
            this->A_is_symmetric,
            vector,
            this->num_rows,
            this->num_columns,
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
/// \sa         cCSCMatrix::dot,
///             cCSCMatrix::transposed_dot
///             cCSCMatrix::transposed_dot_plus

template <typename DataType>
void cCSCMatrix<DataType>::dot_plus(
        const DataType* vector,
        const DataType alpha,
        DataType* product)
{
    cMatrixOperations<DataType>::csc_matvec_plus(
            this->A_data,
            this->A_indices,
            this->A_index_pointer,
            this->A_is_symmetric,
            vector,
            alpha,
            this->num_columns,
            product);
}


// =============
// transpose dot
// =============

/// \brief      Transposed-matrix vector product.
///
/// \details    Performs the matrix vector product \f$ \boldsymbol{y} =
///             \mathbf{A}^{\intercal} \boldsymbol{x} \f$.
///
/// \param[in]  vector
///             A one-dimensional input vector \f$ \boldsymbol{x} \f$ with size
///             the of the number of columns of the matrix \f$ \mathbf{A} \f$.
/// \param[out] product
///             A one-dimensional output vector \f$ \boldsymbol{y} \f$ with the
///             size of the number of rows of \f$ \mathbf{A} \f$. This vector
///             will be overwritten.
///
/// \sa         cCSCMatrix::dot_plus,
///             cCSCMatrix::dot
///             cCSCMatrix::transposed_dot_plus

template <typename DataType>
void cCSCMatrix<DataType>::transpose_dot(
        const DataType* vector,
        DataType* product)
{
    cMatrixOperations<DataType>::csc_transposed_matvec(
            this->A_data,
            this->A_indices,
            this->A_index_pointer,
            vector,
            this->num_columns,
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
/// \sa         cCSCMatrix::dot_plus,
///             cCSCMatrix::transposed_dot
///             cCSCMatrix::dot

template <typename DataType>
void cCSCMatrix<DataType>::transpose_dot_plus(
        const DataType* vector,
        const DataType alpha,
        DataType* product)
{
    cMatrixOperations<DataType>::csc_transposed_matvec_plus(
            this->A_data,
            this->A_indices,
            this->A_index_pointer,
            vector,
            alpha,
            this->num_columns,
            product);
}


// ===============================
// Explicit template instantiation
// ===============================

template class cCSCMatrix<float>;
template class cCSCMatrix<double>;
template class cCSCMatrix<long double>;
