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
#include <omp.h>  // omp_set_num_threads
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
        const FlagType A_is_row_major_,
        const int num_gpu_devices_):

    // Base class constructor
    cLinearOperator<DataType>(num_rows_, num_columns_),
    cDenseMatrix<DataType>(A_, num_rows_, num_columns_, A_is_row_major_),
    cuMatrix<DataType>(num_gpu_devices_),

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
    // Member objects exist if the second constructor was called.
    if (this->copied_host_to_device)
    {
        // Deallocate arrays of data on gpu
        for (int device_id = 0; device_id < this->num_gpu_devices; ++device_id)
        {
            // Switch to a device
            CudaInterface<DataType>::set_device(device_id);

            // Deallocate
            CudaInterface<DataType>::del(this->device_A[device_id]);
        }

        delete[] this->device_A;
        this->device_A = NULL;
    }
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
        // Set the number of threads
        omp_set_num_threads(this->num_gpu_devices);

        // Create array of pointers for data on each gpu device
        this->device_A = new DataType*[this->num_gpu_devices];

        // Size of data
        LongIndexType A_size = this->num_rows * this->num_columns;

        #pragma omp parallel
        {
            // Switch to a device with the same device id as the cpu thread id
            unsigned int thread_id = omp_get_thread_num();
            CudaInterface<DataType>::set_device(thread_id);

            // Allocate device memory and copy data from host
            CudaInterface<DataType>::alloc(this->device_A[thread_id], A_size);
            CudaInterface<DataType>::copy_to_device(this->A, A_size,
                                                    this->device_A[thread_id]);
        }

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

    // Get device id
    int device_id = CudaInterface<DataType>::get_device();

    cuMatrixOperations<DataType>::dense_matvec(
            this->cublas_handle[device_id],
            this->device_A[device_id],
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

    // Get device id
    int device_id = CudaInterface<DataType>::get_device();

    cuMatrixOperations<DataType>::dense_matvec_plus(
            this->cublas_handle[device_id],
            this->device_A[device_id],
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

    // Get device id
    int device_id = CudaInterface<DataType>::get_device();

    cuMatrixOperations<DataType>::dense_transposed_matvec(
            this->cublas_handle[device_id],
            this->device_A[device_id],
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

    // Get device id
    int device_id = CudaInterface<DataType>::get_device();

    cuMatrixOperations<DataType>::dense_transposed_matvec_plus(
            this->cublas_handle[device_id],
            this->device_A[device_id],
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
