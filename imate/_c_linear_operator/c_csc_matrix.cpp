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
#include "../_c_basic_algebra/c_matrix_operations.h"  // cMatrixOperations


// =============
// constructor 1
// =============

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

template <typename DataType>
cCSCMatrix<DataType>::cCSCMatrix(
        const DataType* A_data_,
        const LongIndexType* A_indices_,
        const LongIndexType* A_index_pointer_,
        const LongIndexType num_rows_,
        const LongIndexType num_columns_):

    // Base class constructor
    cLinearOperator<DataType>(num_rows_, num_columns_),

    // Initializer list
    A_data(A_data_),
    A_indices(A_indices_),
    A_index_pointer(A_index_pointer_)
{
}


// ==========
// destructor
// ==========

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

    // Check matrix element-wise
    for (LongIndexType column=0; column < this->num_columns; ++column)
    {
        for (index_pointer=this->A_index_pointer[column];
             index_pointer < this->A_index_pointer[column+1];
             ++index_pointer)
        {
            row = this->A_indices[index_pointer];

            if ((row == column) && \
               (this->A_data[index_pointer] != 1.0))
            {
                matrix_is_identity = 0;
                return matrix_is_identity;
            }
            else if (this->A_data[index_pointer] != 0.0)
            {
                matrix_is_identity = 0;
                return matrix_is_identity;
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

template <typename DataType>
void cCSCMatrix<DataType>::dot(
        const DataType* vector,
        DataType* product)
{
    cMatrixOperations<DataType>::csc_matvec(
            this->A_data,
            this->A_indices,
            this->A_index_pointer,
            vector,
            this->num_rows,
            this->num_columns,
            product);
}


// ========
// dot plus
// ========

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
            vector,
            alpha,
            this->num_columns,
            product);
}


// =============
// transpose dot
// =============

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
