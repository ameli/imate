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

#include "./cu_dense_matrix.h"
#include <cstddef>  // NULL
#include <cassert>  // assert
#include "../_cu_basic_algebra/cu_matrix_operations.h"  // cuMatrixOperations
#include "../_cuda_utilities/cuda_interface.h"  // alloc, copy_to_device, del


// =============
// constructor 1
// =============

template <typename DataType>
cuDenseMatrix<DataType>::cuDenseMatrix():
    device_A(NULL)
{
}


// =============
// constructor 2
// =============

template <typename DataType>
cuDenseMatrix<DataType>::cuDenseMatrix(
        const DataType* A_,
        const LongIndexType num_rows_,
        const LongIndexType num_columns_,
        const FlagType A_is_row_major_):

    // Base class constructor
    cLinearOperator<DataType>(num_rows_, num_columns_),
    cDenseMatrix<DataType>(A_, num_rows_, num_columns_, A_is_row_major_),

    // Initializer list
    device_A(NULL)
{
    this->initialize_cublas_handle();
    this->copy_host_to_device();
}


// ==========
// destructor
// ==========


template <typename DataType>
cuDenseMatrix<DataType>::~cuDenseMatrix()
{
    CudaInterface<DataType>::del(this->device_A);
}


// ===================
// copy host to device
// ===================

/// \brief Copies the member data from the host memory to the device memory.
///

template <typename DataType>
void cuDenseMatrix<DataType>::copy_host_to_device()
{
    if (!this->copied_host_to_device)
    {
        // Allocate device memory and copy data from host
        LongIndexType A_size = this->num_rows * this->num_columns;
        CudaInterface<DataType>::alloc(this->device_A, A_size);
        CudaInterface<DataType>::copy_to_device(this->A, A_size,
                                                this->device_A);

        // Flag to prevent reinitialization
        this->copied_host_to_device = true;
    }
}


// ===
// dot
// ===

template <typename DataType>
void cuDenseMatrix<DataType>::dot(
        const DataType* device_vector,
        DataType* device_product)
{
    assert(this->copied_host_to_device);

    cuMatrixOperations<DataType>::dense_matvec(
            this->cublas_handle,
            this->device_A,
            device_vector,
            this->num_rows,
            this->num_columns,
            this->A_is_row_major,
            device_product);
}


// ========
// dot plus
// ========

template <typename DataType>
void cuDenseMatrix<DataType>::dot_plus(
        const DataType* device_vector,
        const DataType alpha,
        DataType* device_product)
{
    assert(this->copied_host_to_device);

    cuMatrixOperations<DataType>::dense_matvec_plus(
            this->cublas_handle,
            this->device_A,
            device_vector,
            alpha,
            this->num_rows,
            this->num_columns,
            this->A_is_row_major,
            device_product);
}


// =============
// transpose dot
// =============

template <typename DataType>
void cuDenseMatrix<DataType>::transpose_dot(
        const DataType* device_vector,
        DataType* device_product)
{
    assert(this->copied_host_to_device);

    cuMatrixOperations<DataType>::dense_transposed_matvec(
            this->cublas_handle,
            this->device_A,
            device_vector,
            this->num_rows,
            this->num_columns,
            this->A_is_row_major,
            device_product);
}


// ==================
// transpose dot plus
// ==================

template <typename DataType>
void cuDenseMatrix<DataType>::transpose_dot_plus(
        const DataType* device_vector,
        const DataType alpha,
        DataType* device_product)
{
    assert(this->copied_host_to_device);

    cuMatrixOperations<DataType>::dense_transposed_matvec_plus(
            this->cublas_handle,
            this->device_A,
            device_vector,
            alpha,
            this->num_rows,
            this->num_columns,
            this->A_is_row_major,
            device_product);
}


// ===============================
// Explicit template instantiation
// ===============================

template class cuDenseMatrix<float>;
template class cuDenseMatrix<double>;
