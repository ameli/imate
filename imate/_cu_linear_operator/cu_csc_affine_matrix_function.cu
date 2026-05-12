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

#include "./cu_csc_affine_matrix_function.h"
#include <cstddef>  // NULL
#include <cassert>  // assert
#include "../_cu_definitions/cu_types.h" // __nv_fp8_e5m2, __nv_fp8_e4m3,
                                         // __half, __nv_bfloat16
#include "../_definitions/debugging.h"  // ASSERT


// =============
// constructor 1
// =============

/// \brief      Default constructor.
///
/// \details    Matrix \c B is assumed to be the identity matrix.
///
/// \param[in]  A_data_
///             1D array of the data content of sparse matrix. The size of the
///             array is the nnz of the matrix.
/// \param[in]  A_indices_
///             1D array indicating the column of each element in \c A_data_ .
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
/// \param[in]  num_gpu_devices_
///             Number of GPU devices to be utilized for parallel processing.

template <typename DataType>
cuCSCAffineMatrixFunction<DataType>::cuCSCAffineMatrixFunction(
        const DataType* A_data_,
        const LongIndexType* A_indices_,
        const LongIndexType* A_index_pointer_,
        const LongIndexType num_rows_,
        const LongIndexType num_columns_,
        const FlagType A_is_symmetric_,
        const int num_gpu_devices_):

    // Base class constructor
    cLinearOperatorBase(num_rows_, num_columns_),
    cuLinearOperator<DataType>(num_gpu_devices_),

    // Initializer list
    A(A_data_, A_indices_, A_index_pointer_, num_rows_, num_columns_,
      A_is_symmetric_, num_gpu_devices_)
{
    // This constructor is called assuming B is identity
    this->B_is_identity = true;

    // When B is identity, the eigenvalues of A+tB are known for any t
    this->eigenvalue_relation_known = 1;

    // Set gpu device
    this->initialize_cusparse_handle();
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
///             1D array indicating the column of each element in \c A_data_ .
///             The size of this array is the nnz of the matrix.
/// \param[in]  A_index_pointer_
///             1D array pointing to the start of new rows in \c
///             A_indices_ . The size of this array is \c num_rows+1 .
///             The first element of this array is \c 0 and the last element
///             of this array is the nnz of the matrix.
/// \param[in]  num_rows_
///             Number of rows of \c A and \c B
/// \param[in]  num_columns_
///             Number of columns of \c A and \c B
/// \param[in]  A_is_symmetric_
///             Boolean. If \c A is symmetric, set this value to \c 1,
///             otherwise \c 0.
/// \param[in]  B_data_
///             1D array of the data content of sparse matrix. The size of the
///             array is the nnz of the matrix.
/// \param[in]  B_indices_
///             1D array indicating the column of each element in \c B_data_ .
///             The size of this array is the nnz of the matrix.
/// \param[in]  B_index_pointer_
///             1D array pointing to the start of new rows in \c
///             B_indices_ . The size of this array is \c num_rows+1 .
///             The first element of this array is \c 0 and the last element
///             of this array is the nnz of the matrix.
/// \param[in]  B_is_symmetric_
///             Boolean. If \c B is symmetric, set this value to \c 1,
///             otherwise \c 0.
/// \param[in]  num_gpu_devices_
///             Number of GPU devices to be utilized for parallel processing.

template <typename DataType>
cuCSCAffineMatrixFunction<DataType>::cuCSCAffineMatrixFunction(
        const DataType* A_data_,
        const LongIndexType* A_indices_,
        const LongIndexType* A_index_pointer_,
        const LongIndexType num_rows_,
        const LongIndexType num_columns_,
        const FlagType A_is_symmetric_,
        const DataType* B_data_,
        const LongIndexType* B_indices_,
        const LongIndexType* B_index_pointer_,
        const FlagType B_is_symmetric_,
        const int num_gpu_devices_):

    // Base class constructor
    cLinearOperatorBase(num_rows_, num_columns_),
    cuLinearOperator<DataType>(num_gpu_devices_),

    // Initializer list
    A(A_data_, A_indices_, A_index_pointer_, num_rows_, num_columns_,
      A_is_symmetric_, num_gpu_devices_),
    B(B_data_, B_indices_, B_index_pointer_, num_rows_, num_columns_,
      B_is_symmetric_, num_gpu_devices_)
{
    // Matrix B is assumed to be non-zero. Check if it is identity or generic
    if (this->B.is_identity_matrix())
    {
        this->B_is_identity = true;
        this->eigenvalue_relation_known = 1;
    }

    // Set gpu device
    this->initialize_cusparse_handle();
}


// ==========
// destructor
// ==========

/// \brief Destructor.
///

template <typename DataType>
cuCSCAffineMatrixFunction<DataType>::~cuCSCAffineMatrixFunction()
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
void cuCSCAffineMatrixFunction<DataType>::set_symmetry(
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
///             This array should be on GPU device.
/// \param[out] product
///             A one-dimensional output vector \f$ \boldsymbol{y} \f$ with the
///             size of the number of rows of \f$ \mathbf{A} \f$. This vector
///             will be overwritten. This array should be on GPU device.
///
/// \sa         cuCSCAffineMatrixFunction::transpose_dot

template <typename DataType>
void cuCSCAffineMatrixFunction<DataType>::dot(
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
        ASSERT((this->parameters != NULL), "Parameter is not set.");

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
        ASSERT((this->parameters != NULL), "Parameter is not set.");

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
///             This array should be on GPU device.
/// \param[out] product
///             A one-dimensional output vector \f$ \boldsymbol{y} \f$ with the
///             size of the number of rows of \f$ \mathbf{A} \f$. This array
///             should be on GPU device.
///
/// \sa         cuCSCAffineMatrixFunction::dot

template <typename DataType>
void cuCSCAffineMatrixFunction<DataType>::transpose_dot(
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
        ASSERT((this->parameters != NULL), "Parameter is not set.");

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
        ASSERT((this->parameters != NULL), "Parameter is not set.");

        // Adding "parameter * B * input vector" to the product
        this->B.transpose_dot_plus(vector, this->parameters[0], product);
    }
}


// ===============================
// Explicit template instantiation
// ===============================

#if defined(USE_CUDA_FP8_E5M2) && (USE_CUDA_FP8_E5M2 == 1)
    template class cuCSCAffineMatrixFunction<__nv_fp8_e5m2>;
#endif

#if defined(USE_CUDA_FP8_E4M3) && (USE_CUDA_FP8_E4M3 == 1)
    template class cuCSCAffineMatrixFunction<__nv_fp8_e4m3>;
#endif

#if defined(USE_CUDA_FP16) && (USE_CUDA_FP16 == 1)
    template class cuCSCAffineMatrixFunction<__half>;
#endif

#if defined(USE_CUDA_BF16) && (USE_CUDA_BF16 == 1)
    template class cuCSCAffineMatrixFunction<__nv_bfloat16>;
#endif

#if defined(USE_CUDA_FP32) && (USE_CUDA_FP32 == 1)
    template class cuCSCAffineMatrixFunction<float>;
#endif

#if defined(USE_CUDA_FP64) && (USE_CUDA_FP64 == 1)
    template class cuCSCAffineMatrixFunction<double>;
#endif
