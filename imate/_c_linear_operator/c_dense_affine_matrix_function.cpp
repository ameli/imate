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

#include "./c_dense_affine_matrix_function.h"
#include <cassert>  // assert
#include <cstddef>  // NULL


// =============
// constructor 1
// =============

/// \brief      Default constructor.
///
/// \details    Matrix \c B is assumed to be the identity matrix.
///
/// \param[in]  A_
///             1D array that represents a 2D dense array with either C (row)
///             major ordering or Fortran (column) major ordering. The major
///             ordering should de defined by \c A_is_row_major flag.
/// \param[in]  num_rows_
///             Number of rows of \c A and \c B
/// \param[in]  num_columns_
///             Number of columns of \c A and \c B
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
cDenseAffineMatrixFunction<DataType>::cDenseAffineMatrixFunction(
        const DataType* A_,
        const LongIndexType num_rows_,
        const LongIndexType num_columns_,
        const FlagType A_is_row_major_,
        const FlagType A_is_symmetric_):

    // Base class constructor
    cLinearOperatorBase(num_rows_, num_columns_),

    // Initializer list
    A(A_, num_rows_, num_columns_, A_is_row_major_, A_is_symmetric_)
{
    // This constructor is called assuming B is identity
    this->B_is_identity = true;

    // When B is identity, the eigenvalues of A+tB are known for any t
    this->eigenvalue_relation_known = 1;
}


// =============
// constructor 2
// =============

/// \brief      Constructor.
///
/// \details    Matrix \c B is assumed to be the identity matrix.
///
/// \param[in]  A_
///             1D array that represents a 2D dense array with either C (row)
///             major ordering or Fortran (column) major ordering. The major
///             ordering should de defined by \c A_is_row_major flag.
/// \param[in]  num_rows_
///             Number of rows of \c A and \c B
/// \param[in]  num_columns_
///             Number of columns of \c A and \c B
/// \param[in]  A_is_row_major_
///             Boolean, can be \c 0 or \c 1 as follows:
///             * If \c A is row major (C ordering where the last index is
///               contiguous) this value should be \c 1.
///             * If \c A is column major (Fortran ordering where the first
///               index is contiguous), this value should be set to \c 0.
/// \param[in]  A_is_symmetric_
///             Boolean. If \c A is symmetric, set this value to \c 1,
///             otherwise \c 0.
/// \param[in]  B_
///             1D array that represents a 2D dense array with either C (row)
///             major ordering or Fortran (column) major ordering. The major
///             ordering should de defined by \c A_is_row_major flag.
/// \param[in]  B_is_row_major_
///             Boolean, can be \c 0 or \c 1 as follows:
///             * If \c B is row major (C ordering where the last index is
///               contiguous) this value should be \c 1.
///             * If \c B is column major (Fortran ordering where the first
///               index is contiguous), this value should be set to \c 0.
/// \param[in]  B_is_symmetric_
///             Boolean. If \c B is symmetric, set this value to \c 1,
///             otherwise \c 0.

template <typename DataType>
cDenseAffineMatrixFunction<DataType>::cDenseAffineMatrixFunction(
        const DataType* A_,
        const LongIndexType num_rows_,
        const LongIndexType num_columns_,
        const FlagType A_is_row_major_,
        const FlagType A_is_symmetric_,
        const DataType* B_,
        const FlagType B_is_row_major_,
        const FlagType B_is_symmetric_):

    // Base class constructor
    cLinearOperatorBase(num_rows_, num_columns_),

    // Initializer list
    A(A_, num_rows_, num_columns_, A_is_row_major_, A_is_symmetric_),
    B(B_, num_rows_, num_columns_, B_is_row_major_, B_is_symmetric_)
{
    // Matrix B is assumed to be non-zero. Check if it is identity or generic
    if (this->B.is_identity_matrix())
    {
        this->B_is_identity = true;
        this->eigenvalue_relation_known = 1;
    }
}


// ==========
// destructor
// ==========

/// \brief Destructor.
/// 

template <typename DataType>
cDenseAffineMatrixFunction<DataType>::~cDenseAffineMatrixFunction()
{
}


// ============
// set symmetry
// ============

/// \brief     Specify whether the matrices are symmetic or non-symmetric.
///
/// \details   This function overwrites the symmetry status that has been set
///            by the constructor. Note that the symmetry status of both
///            matrices \f$ \mathbf{A} \f$ and \f$ \mathbf{B} \f$ in the
///            linear operator \f$ \mathbf{A} + t \mathbf{B} \f$ will be set
///            together.
///
/// \param[in] symmetric
///            Boolean. If set to \c 1, the matrix is assumed to be symmetric.
///            Otherwiese non-symmetric.

template <typename DataType>
void cDenseAffineMatrixFunction<DataType>::set_symmetry(
        const FlagType symmetric)
{
    if (symmetric == 1)
    {
        this->A.set_symmetry(1);
        this->B.set_symmetry(1);
    }
    else
    {
        this->A.set_symmetry(0);
        this->B.set_symmetry(0);
    }
}


// ===
// dot
// ===

/// \brief      Matrix vector product.
///
/// \details    Performs the matrix vector product \f$ \boldsymbol{y} =
///             (\mathbf{A} + t \mathbf{B}) \boldsymbol{x} \f$.
///
/// \param[in]  vector
///             A one-dimensional input vector \f$ \boldsymbol{x} \f$ with size
///             the of the number of columns of the matrix \f$ \mathbf{A} \f$.
/// \param[out] product
///             A one-dimensional output vector \f$ \boldsymbol{y} \f$ with the
///             size of the number of rows of \f$ \mathbf{A} \f$. This vector
///             will be overwritten.
///
/// \sa         cDenseAffineMatrixFunction::transpose_dot

template <typename DataType>
void cDenseAffineMatrixFunction<DataType>::dot(
        const DataType* vector,
        DataType* product)
{
    // Matrix A times vector
    this->A.dot(vector, product);
    LongIndexType min_vector_size;

    // Matrix B times vector to be added to the product
    if (this->B_is_identity)
    {
        // Check parameter is set
        assert((this->parameters != NULL) && "Parameter is not set.");

        // Find minimum of the number of rows and columns
        min_vector_size = \
            (this->num_rows < this->num_columns) ? \
            this->num_rows : this->num_columns;

        // Adding input vector to product
        this->_add_scaled_vector(vector, min_vector_size,
                                 this->parameters[0], product);
    }
    else
    {
        // Check parameter is set
        assert((this->parameters != NULL) && "Parameter is not set.");

        // Adding parameter times B times input vector to the product
        this->B.dot_plus(vector, this->parameters[0], product);
    }
}


// =============
// transpose dot
// =============

/// \brief      Matrix vector product written in place.
///
/// \details    Performs the matrix vector product \f$ \boldsymbol{y} =
///             (\mathbf{A} + t \mathbf{B})^{\intercal} \boldsymbol{x} \f$.
///
/// \param[in]  vector
///             A one-dimensional input vector \f$ \boldsymbol{x} \f$ with size
///             the of the number of columns of the matrix \f$ \mathbf{A} \f$.
/// \param[out] product
///             A one-dimensional output vector \f$ \boldsymbol{y} \f$ with the
///             size of the number of rows of \f$ \mathbf{A} \f$.
///
/// \sa         cDenseAffineMatrixFunction::dot

template <typename DataType>
void cDenseAffineMatrixFunction<DataType>::transpose_dot(
        const DataType* vector,
        DataType* product)
{
    // Matrix A times vector
    this->A.transpose_dot(vector, product);
    LongIndexType min_vector_size;

    // Matrix B times vector to be added to the product
    if (this->B_is_identity)
    {
        // Check parameter is set
        assert((this->parameters != NULL) && "Parameter is not set.");

        // Find minimum of the number of rows and columns
        min_vector_size = \
            (this->num_rows < this->num_columns) ? \
            this->num_rows : this->num_columns;

        // Adding input vector to product
        this->_add_scaled_vector(vector, min_vector_size,
                                 this->parameters[0], product);
    }
    else
    {
        // Check parameter is set
        assert((this->parameters != NULL) && "Parameter is not set.");

        // Adding "parameter * B * input vector" to the product
        this->B.transpose_dot_plus(vector, this->parameters[0], product);
    }
}


// ===============================
// Explicit template instantiation
// ===============================

template class cDenseAffineMatrixFunction<float>;
template class cDenseAffineMatrixFunction<double>;
template class cDenseAffineMatrixFunction<long double>;
