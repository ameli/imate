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
#include "../_c_basic_algebra/c_matrix_operations.h"  // cMatrixOperations


// =============
// constructor 1
// =============

template <typename DataType>
cDenseMatrix<DataType>::cDenseMatrix():
    A(NULL),
    A_is_row_major(0)
{
}


// =============
// constructor 2
// =============

template <typename DataType>
cDenseMatrix<DataType>::cDenseMatrix(
        const DataType* A_,
        const LongIndexType num_rows_,
        const LongIndexType num_columns_,
        const FlagType A_is_row_major_):

    // Base class constructor
    cLinearOperator<DataType>(num_rows_, num_columns_),

    // Initializer list
    A(A_),
    A_is_row_major(A_is_row_major_)
{
}


// ==========
// destructor
// ==========


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

    // Check matrix element-wise
    for (LongIndexType row=0; row < this->num_rows; ++row)
    {
        for (LongIndexType column=0; column < this-> num_columns; ++column)
        {
            // Get an element of the matrix
            if (this->A_is_row_major)
            {
                matrix_element = this->A[row * this->num_columns + column];
            }
            else
            {
                matrix_element = this->A[column * this->num_rows + row];
            }

            // Check the value of element with identity matrix
            if ((row == column) && (matrix_element != 1.0))
            {
                matrix_is_identity = 0;
                return matrix_is_identity;
            }
            else if (matrix_element != 0.0)
            {
                matrix_is_identity = 0;
                return matrix_is_identity;
            }
        }
    }

    return matrix_is_identity;
}


// ===
// dot
// ===

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
            product);
}


// ========
// dot plus
// ========

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
            product);
}


// =============
// transpose dot
// =============

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
            product);
}


// ==================
// transpose dot plus
// ==================

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
            product);
}


// ===============================
// Explicit template instantiation
// ===============================

template class cDenseMatrix<float>;
template class cDenseMatrix<double>;
template class cDenseMatrix<long double>;
