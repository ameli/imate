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

#include "./cu_csr_matrix.h"
#include <omp.h>  // omp_set_num_threads
#include <cstddef>  // NULL
#include <cassert>  // assert
#include "../_cu_basic_algebra/cu_matrix_operations.h"  // cuMatrixOperations
#include "../_cu_basic_algebra/cusparse_interface.h"  // cusparse_interface
#include "../_cuda_utilities/cuda_interface.h"  // CudaInterface


// =============
// constructor 1
// =============

/// \brief Default constructor
///

template <typename DataType>
cuCSRMatrix<DataType>::cuCSRMatrix():
    device_A_data(NULL),
    device_A_indices(NULL),
    device_A_index_pointer(NULL),
    device_buffer(NULL),
    device_buffer_num_bytes(NULL),
    cusparse_matrix_A(NULL)
{
}


// =============
// constructor 2
// =============

/// \brief Constructor with arguments.
///

template <typename DataType>
cuCSRMatrix<DataType>::cuCSRMatrix(
        const DataType* A_data_,
        const LongIndexType* A_indices_,
        const LongIndexType* A_index_pointer_,
        const LongIndexType num_rows_,
        const LongIndexType num_columns_,
        const int num_gpu_devices_):

    // Base class constructor
    cLinearOperator<DataType>(num_rows_, num_columns_),
    cCSRMatrix<DataType>(A_data_, A_indices_, A_index_pointer_, num_rows_,
                         num_columns_),
    cuMatrix<DataType>(num_gpu_devices_),

    // Initializer list
    device_A_data(NULL),
    device_A_indices(NULL),
    device_A_index_pointer(NULL),
    device_buffer(NULL),
    cusparse_matrix_A(NULL)
{
    this->initialize_cusparse_handle();
    this->copy_host_to_device();

    // Initialize device buffer
    this->device_buffer = new void*[this->num_gpu_devices];
    this->device_buffer_num_bytes = new size_t[this->num_gpu_devices];
    for (int device_id=0; device_id < this->num_gpu_devices; ++device_id)
    {
        this->device_buffer[device_id] = NULL;
        this->device_buffer_num_bytes[device_id] = 0;
    }
}


// ==========
// destructor
// ==========

/// \brief Virtual desructor.
///

template <typename DataType>
cuCSRMatrix<DataType>::~cuCSRMatrix()
{
    // Member objects exist if the second constructor was called.
    if (this->copied_host_to_device)
    {
        // Deallocate arrays of data on gpu
        for (int device_id=0; device_id < this->num_gpu_devices; ++device_id)
        {
            // Switch to a device
            CudaInterface<DataType>::set_device(device_id);

            // Deallocate
            CudaInterface<DataType>::del(this->device_A_data[device_id]);
            CudaInterface<LongIndexType>::del(
                    this->device_A_indices[device_id]);
            CudaInterface<LongIndexType>::del(
                    this->device_A_index_pointer[device_id]);
            CudaInterface<LongIndexType>::del(this->device_buffer[device_id]);
            cusparse_interface::destroy_cusparse_matrix(
                    this->cusparse_matrix_A[device_id]);
        }
    }

    // Deallocate arrays of pointers on cpu
    if (this->device_A_data != NULL)
    {
        delete[] this->device_A_data;
        this->device_A_data = NULL;
    }

    if (this->device_A_indices != NULL)
    {
        delete[] this->device_A_indices;
        this->device_A_indices = NULL;
    }

    if (this->device_A_index_pointer != NULL)
    {
        delete[] this->device_A_index_pointer;
        this->device_A_index_pointer = NULL;
    }

    if (this->device_buffer != NULL)
    {
        delete[] this->device_buffer;
        this->device_buffer = NULL;
    }

    if (this->device_buffer_num_bytes != NULL)
    {
        delete[] this->device_buffer_num_bytes;
        this->device_buffer_num_bytes = NULL;
    }

    if (this->cusparse_matrix_A != NULL)
    {
        delete[] this->cusparse_matrix_A;
        this->cusparse_matrix_A = NULL;
    }
}


// ===================
// copy host to device
// ===================

/// \brief Copies the member data from the host memory to the device memory.
///

template <typename DataType>
void cuCSRMatrix<DataType>::copy_host_to_device()
{
    if (!this->copied_host_to_device)
    {
        // Set the number of threads
        omp_set_num_threads(this->num_gpu_devices);

        // Array sizes
        LongIndexType A_data_size = this->get_nnz();
        LongIndexType A_indices_size = A_data_size;
        LongIndexType A_index_pointer_size = this->num_rows + 1;
        LongIndexType A_nnz = this->get_nnz();

        // Create array of pointers for data on each gpu device
        this->device_A_data = new DataType*[this->num_gpu_devices];
        this->device_A_indices = new LongIndexType*[this->num_gpu_devices];
        this->device_A_index_pointer = \
            new LongIndexType*[this->num_gpu_devices];
        this->cusparse_matrix_A = \
            new cusparseSpMatDescr_t[this->num_gpu_devices];

        #pragma omp parallel
        {
            // Switch to a device with the same device id as the cpu thread id
            unsigned int thread_id = omp_get_thread_num();
            CudaInterface<DataType>::set_device(thread_id);

            // A_data
            CudaInterface<DataType>::alloc(this->device_A_data[thread_id],
                                           A_data_size);
            CudaInterface<DataType>::copy_to_device(
                    this->A_data, A_data_size, this->device_A_data[thread_id]);

            // A_indices
            CudaInterface<LongIndexType>::alloc(
                    this->device_A_indices[thread_id], A_indices_size);
            CudaInterface<LongIndexType>::copy_to_device(
                    this->A_indices, A_indices_size,
                    this->device_A_indices[thread_id]);

            // A_index_pointer
            CudaInterface<LongIndexType>::alloc(
                    this->device_A_index_pointer[thread_id],
                    A_index_pointer_size);
            CudaInterface<LongIndexType>::copy_to_device(
                    this->A_index_pointer, A_index_pointer_size,
                    this->device_A_index_pointer[thread_id]);

            // Create cusparse matrix
            cusparse_interface::create_cusparse_matrix(
                    this->cusparse_matrix_A[thread_id], this->num_rows,
                    this->num_columns, A_nnz, this->device_A_data[thread_id],
                    this->device_A_indices[thread_id],
                    this->device_A_index_pointer[thread_id]);
        }

        // Flag to prevent reinitialization
        this->copied_host_to_device = true;
    }
}


// ===============
// allocate buffer
// ===============

/// \brief   Allocates an external buffer for matrix-vector multiplication
///          using \c cusparseSpMV function.
///
/// \details If buffer size if not the same as required buffer size, allocate
///          (or reallocate) memory. The allocation is always performed in the
///          fisr call of this function since buffer size is initialized to
///          zero in constructor. But for the next calls it might not be
///          reallocated if the buffer size is the same.

template <typename DataType>
void cuCSRMatrix<DataType>::allocate_buffer(
        const int device_id,
        cusparseOperation_t cusparse_operation,
        const DataType alpha,
        const DataType beta,
        cusparseDnVecDescr_t& cusparse_input_vector,
        cusparseDnVecDescr_t& cusparse_output_vector,
        cusparseSpMVAlg_t algorithm)
{
    // Find the buffer size needed for matrix-vector multiplication
    size_t required_buffer_size;
    cusparse_interface::cusparse_matrix_buffer_size(
            this->cusparse_handle[device_id], cusparse_operation, alpha,
            this->cusparse_matrix_A[device_id], cusparse_input_vector, beta,
            cusparse_output_vector, algorithm, &required_buffer_size);

    if (this->device_buffer_num_bytes[device_id] != required_buffer_size)
    {
        // Update the buffer size
        this->device_buffer_num_bytes[device_id] = required_buffer_size;

        // Delete buffer if it was allocated previously
        CudaInterface<DataType>::del(this->device_buffer[device_id]);

        // Allocate (or reallocate) buffer on device.
        CudaInterface<DataType>::alloc_bytes(
                this->device_buffer[device_id],
                this->device_buffer_num_bytes[device_id]);
    }
}


// ===
// dot
// ===

template <typename DataType>
void cuCSRMatrix<DataType>::dot(
        const DataType* device_vector,
        DataType* device_product)
{
    assert(this->copied_host_to_device);

    // Create cusparse vector for the input vector
    cusparseDnVecDescr_t cusparse_input_vector;
    cusparse_interface::create_cusparse_vector(
            cusparse_input_vector, this->num_columns,
            const_cast<DataType*>(device_vector));

    // Create cusparse vector for the output vector
    cusparseDnVecDescr_t cusparse_output_vector;
    cusparse_interface::create_cusparse_vector(
            cusparse_output_vector, this->num_rows, device_product);

    // Matrix vector settings
    DataType alpha = 1.0;
    DataType beta = 0.0;
    cusparseOperation_t cusparse_operation = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseSpMVAlg_t algorithm = CUSPARSE_MV_ALG_DEFAULT;

    // Get device id
    int device_id = CudaInterface<DataType>::get_device();

    // Allocate device buffer (or reallocation if needed)
    this->allocate_buffer(device_id, cusparse_operation, alpha, beta,
                          cusparse_input_vector, cusparse_output_vector,
                          algorithm);

    // Matrix vector multiplication
    cusparse_interface::cusparse_matvec(
            this->cusparse_handle[device_id], cusparse_operation, alpha,
            this->cusparse_matrix_A[device_id], cusparse_input_vector, beta,
            cusparse_output_vector, algorithm, this->device_buffer[device_id]);

    // Destroy cusparse vectors
    cusparse_interface::destroy_cusparse_vector(cusparse_input_vector);
    cusparse_interface::destroy_cusparse_vector(cusparse_output_vector);
}


// ========
// dot plus
// ========

template <typename DataType>
void cuCSRMatrix<DataType>::dot_plus(
        const DataType* device_vector,
        const DataType alpha,
        DataType* device_product)
{
    assert(this->copied_host_to_device);

    // Create cusparse vector for the input vector
    cusparseDnVecDescr_t cusparse_input_vector;
    cusparse_interface::create_cusparse_vector(
            cusparse_input_vector, this->num_columns,
            const_cast<DataType*>(device_vector));

    // Create cusparse vector for the output vector
    cusparseDnVecDescr_t cusparse_output_vector;
    cusparse_interface::create_cusparse_vector(
            cusparse_output_vector, this->num_rows, device_product);

    // Matrix vector settings
    DataType beta = 1.0;
    cusparseOperation_t cusparse_operation = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseSpMVAlg_t algorithm = CUSPARSE_MV_ALG_DEFAULT;

    // Get device id
    int device_id = CudaInterface<DataType>::get_device();

    // Allocate device buffer (or reallocation if needed)
    this->allocate_buffer(device_id, cusparse_operation, alpha, beta,
                          cusparse_input_vector, cusparse_output_vector,
                          algorithm);

    // Matrix vector multiplication
    cusparse_interface::cusparse_matvec(
            this->cusparse_handle[device_id], cusparse_operation, alpha,
            this->cusparse_matrix_A[device_id], cusparse_input_vector, beta,
            cusparse_output_vector, algorithm, this->device_buffer[device_id]);

    // Destroy cusparse vectors
    cusparse_interface::destroy_cusparse_vector(cusparse_input_vector);
    cusparse_interface::destroy_cusparse_vector(cusparse_output_vector);
}


// =============
// transpose dot
// =============

template <typename DataType>
void cuCSRMatrix<DataType>::transpose_dot(
        const DataType* device_vector,
        DataType* device_product)
{
    assert(this->copied_host_to_device);

    // Create cusparse vector for the input vector
    cusparseDnVecDescr_t cusparse_input_vector;
    cusparse_interface::create_cusparse_vector(
            cusparse_input_vector, this->num_columns,
            const_cast<DataType*>(device_vector));

    // Create cusparse vector for the output vector
    cusparseDnVecDescr_t cusparse_output_vector;
    cusparse_interface::create_cusparse_vector(
            cusparse_output_vector, this->num_rows, device_product);

    // Matrix vector settings
    DataType alpha = 1.0;
    DataType beta = 0.0;
    cusparseOperation_t cusparse_operation = CUSPARSE_OPERATION_TRANSPOSE;
    cusparseSpMVAlg_t algorithm = CUSPARSE_MV_ALG_DEFAULT;

    // Get device id
    int device_id = CudaInterface<DataType>::get_device();

    // Allocate device buffer (or reallocation if needed)
    this->allocate_buffer(device_id, cusparse_operation, alpha, beta,
                          cusparse_input_vector, cusparse_output_vector,
                          algorithm);

    // Matrix vector multiplication
    cusparse_interface::cusparse_matvec(
            this->cusparse_handle[device_id], cusparse_operation, alpha,
            this->cusparse_matrix_A[device_id], cusparse_input_vector, beta,
            cusparse_output_vector, algorithm, this->device_buffer[device_id]);

    // Destroy cusparse vectors
    cusparse_interface::destroy_cusparse_vector(cusparse_input_vector);
    cusparse_interface::destroy_cusparse_vector(cusparse_output_vector);
}


// ==================
// transpose dot plus
// ==================

template <typename DataType>
void cuCSRMatrix<DataType>::transpose_dot_plus(
        const DataType* device_vector,
        const DataType alpha,
        DataType* device_product)
{
    assert(this->copied_host_to_device);

    // Create cusparse vector for the input vector
    cusparseDnVecDescr_t cusparse_input_vector;
    cusparse_interface::create_cusparse_vector(
            cusparse_input_vector, this->num_columns,
            const_cast<DataType*>(device_vector));

    // Create cusparse vector for the output vector
    cusparseDnVecDescr_t cusparse_output_vector;
    cusparse_interface::create_cusparse_vector(
            cusparse_output_vector, this->num_rows, device_product);

    // Matrix vector settings
    DataType beta = 1.0;
    cusparseOperation_t cusparse_operation = CUSPARSE_OPERATION_TRANSPOSE;
    cusparseSpMVAlg_t algorithm = CUSPARSE_MV_ALG_DEFAULT;

    // Get device id
    int device_id = CudaInterface<DataType>::get_device();

    // Allocate device buffer (or reallocation if needed)
    this->allocate_buffer(device_id, cusparse_operation, alpha, beta,
                          cusparse_input_vector, cusparse_output_vector,
                          algorithm);

    // Matrix vector multiplication
    cusparse_interface::cusparse_matvec(
            this->cusparse_handle[device_id], cusparse_operation, alpha,
            this->cusparse_matrix_A[device_id], cusparse_input_vector, beta,
            cusparse_output_vector, algorithm, this->device_buffer[device_id]);

    // Destroy cusparse vectors
    cusparse_interface::destroy_cusparse_vector(cusparse_input_vector);
    cusparse_interface::destroy_cusparse_vector(cusparse_output_vector);
}


// ===============================
// Explicit template instantiation
// ===============================

template class cuCSRMatrix<float>;
template class cuCSRMatrix<double>;
