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

#include "./cu_csc_matrix.h"
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
cuCSCMatrix<DataType>::cuCSCMatrix():
    device_A_data(NULL),
    device_A_indices(NULL),
    device_A_index_pointer(NULL),
    device_buffer(NULL),
    device_buffer_num_bytes(0)
{
}


// =============
// constructor 2
// =============

/// \brief Constructor with arguments.
///

template <typename DataType>
cuCSCMatrix<DataType>::cuCSCMatrix(
        const DataType* A_data_,
        const LongIndexType* A_indices_,
        const LongIndexType* A_index_pointer_,
        const LongIndexType num_rows_,
        const LongIndexType num_columns_):

    // Base class constructor
    cLinearOperator<DataType>(num_rows_, num_columns_),
    cCSCMatrix<DataType>(A_data_, A_indices_, A_index_pointer_, num_rows_,
                         num_columns_),

    // Initializer list
    device_A_data(NULL),
    device_A_indices(NULL),
    device_A_index_pointer(NULL),
    device_buffer(NULL),
    device_buffer_num_bytes(0)
{
    this->initialize_cusparse_handle();
    this->copy_host_to_device();
}


// ==========
// destructor
// ==========

/// \brief Virtual desructor.
///

template <typename DataType>
cuCSCMatrix<DataType>::~cuCSCMatrix()
{
    // Device arrays
    CudaInterface<DataType>::del(this->device_A_data);
    CudaInterface<LongIndexType>::del(this->device_A_indices);
    CudaInterface<LongIndexType>::del(this->device_A_index_pointer);
    CudaInterface<LongIndexType>::del(this->device_buffer);

    // Cusparse matrix object exists only if the second constructor was called
    if (this->copied_host_to_device)
    {
        cusparse_interface::destroy_cusparse_matrix(this->cusparse_matrix_A);
    }
}


// ===================
// copy host to device
// ===================

/// \brief Copies the member data from the host memory to the device memory.
///
/// \note  Despite the input matrix is a CSC matrix, we treat it as a CSR
///        matrix, since cusparse's interface is only for CSR matrices. For
///        this, we swap the number of columns and rows from the input matrix
///        to the cusparse matrix.

template <typename DataType>
void cuCSCMatrix<DataType>::copy_host_to_device()
{
    if (!this->copied_host_to_device)
    {
        // A_data
        LongIndexType A_data_size = this->get_nnz();
        CudaInterface<DataType>::alloc(this->device_A_data, A_data_size);
        CudaInterface<DataType>::copy_to_device(this->A_data, A_data_size,
                                                this->device_A_data);

        // A_indices
        LongIndexType A_indices_size = A_data_size;
        CudaInterface<LongIndexType>::alloc(
                this->device_A_indices, A_indices_size);
        CudaInterface<LongIndexType>::copy_to_device(
                this->A_indices, A_indices_size, this->device_A_indices);

        // A_index_pointer
        LongIndexType A_index_pointer_size = this->num_rows + 1;
        CudaInterface<LongIndexType>::alloc(
                this->device_A_index_pointer, A_index_pointer_size);
        CudaInterface<LongIndexType>::copy_to_device(
                this->A_index_pointer, A_index_pointer_size,
                this->device_A_index_pointer);

        // Create cusparse matrix
        LongIndexType A_nnz = this->get_nnz();

        // Swapping the number of rows and columns to treat the input CSC
        // matrix as a CSR matrix.
        LongIndexType csc_num_rows = this->num_columns;
        LongIndexType csc_num_columns = this->num_rows;

        cusparse_interface::create_cusparse_matrix(
                this->cusparse_matrix_A, csc_num_rows, csc_num_columns,
                A_nnz, this->device_A_data, this->device_A_indices,
                this->device_A_index_pointer);

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
void cuCSCMatrix<DataType>::allocate_buffer(
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
            this->cusparse_handle, cusparse_operation, alpha,
            this->cusparse_matrix_A, cusparse_input_vector, beta,
            cusparse_output_vector, algorithm, &required_buffer_size);

    if (this->device_buffer_num_bytes != required_buffer_size)
    {
        // Update the buffer size
        this->device_buffer_num_bytes = required_buffer_size;

        // Delete buffer if it was allocated previously
        CudaInterface<DataType>::del(this->device_buffer);

        // Allocate (or reallocate) buffer on device.
        CudaInterface<DataType>::alloc_bytes(this->device_buffer,
                                             this->device_buffer_num_bytes);
    }
}


// ===
// dot
// ===

template <typename DataType>
void cuCSCMatrix<DataType>::dot(
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

    // Using transpose operation since we treat CSC matrix as CSR
    cusparseOperation_t cusparse_operation = CUSPARSE_OPERATION_TRANSPOSE;
    cusparseSpMVAlg_t algorithm = CUSPARSE_MV_ALG_DEFAULT;

    // Allocate device buffer (or reallocation if needed)
    this->allocate_buffer(cusparse_operation, alpha, beta,
                          cusparse_input_vector, cusparse_output_vector,
                          algorithm);

    // Matrix vector multiplication
    cusparse_interface::cusparse_matvec(
            this->cusparse_handle, cusparse_operation, alpha,
            this->cusparse_matrix_A, cusparse_input_vector, beta,
            cusparse_output_vector, algorithm, this->device_buffer);

    // Destroy cusparse vectors
    cusparse_interface::destroy_cusparse_vector(cusparse_input_vector);
    cusparse_interface::destroy_cusparse_vector(cusparse_output_vector);
}


// ========
// dot plus
// ========

template <typename DataType>
void cuCSCMatrix<DataType>::dot_plus(
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

    // Using transpose operation since we treat CSC matrix as CSR
    cusparseOperation_t cusparse_operation = CUSPARSE_OPERATION_TRANSPOSE;
    cusparseSpMVAlg_t algorithm = CUSPARSE_MV_ALG_DEFAULT;

    // Allocate device buffer (or reallocation if needed)
    this->allocate_buffer(cusparse_operation, alpha, beta,
                          cusparse_input_vector, cusparse_output_vector,
                          algorithm);

    // Matrix vector multiplication
    cusparse_interface::cusparse_matvec(
            this->cusparse_handle, cusparse_operation, alpha,
            this->cusparse_matrix_A, cusparse_input_vector, beta,
            cusparse_output_vector, algorithm, this->device_buffer);

    // Destroy cusparse vectors
    cusparse_interface::destroy_cusparse_vector(cusparse_input_vector);
    cusparse_interface::destroy_cusparse_vector(cusparse_output_vector);
}


// =============
// transpose dot
// =============

template <typename DataType>
void cuCSCMatrix<DataType>::transpose_dot(
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

    // Using non-transpose operation since we treat CSC matrix as CSR
    cusparseOperation_t cusparse_operation = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseSpMVAlg_t algorithm = CUSPARSE_MV_ALG_DEFAULT;

    // Allocate device buffer (or reallocation if needed)
    this->allocate_buffer(cusparse_operation, alpha, beta,
                          cusparse_input_vector, cusparse_output_vector,
                          algorithm);

    // Matrix vector multiplication
    cusparse_interface::cusparse_matvec(
            this->cusparse_handle, cusparse_operation, alpha,
            this->cusparse_matrix_A, cusparse_input_vector, beta,
            cusparse_output_vector, algorithm, this->device_buffer);

    // Destroy cusparse vectors
    cusparse_interface::destroy_cusparse_vector(cusparse_input_vector);
    cusparse_interface::destroy_cusparse_vector(cusparse_output_vector);
}


// ==================
// transpose dot plus
// ==================

template <typename DataType>
void cuCSCMatrix<DataType>::transpose_dot_plus(
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

    // Using non-transpose operation since we treat CSC matrix as CSR
    cusparseOperation_t cusparse_operation = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseSpMVAlg_t algorithm = CUSPARSE_MV_ALG_DEFAULT;

    // Allocate device buffer (or reallocation if needed)
    this->allocate_buffer(cusparse_operation, alpha, beta,
                          cusparse_input_vector, cusparse_output_vector,
                          algorithm);

    // Matrix vector multiplication
    cusparse_interface::cusparse_matvec(
            this->cusparse_handle, cusparse_operation, alpha,
            this->cusparse_matrix_A, cusparse_input_vector, beta,
            cusparse_output_vector, algorithm, this->device_buffer);

    // Destroy cusparse vectors
    cusparse_interface::destroy_cusparse_vector(cusparse_input_vector);
    cusparse_interface::destroy_cusparse_vector(cusparse_output_vector);
}


// ===============================
// Explicit template instantiation
// ===============================

template class cuCSCMatrix<float>;
template class cuCSCMatrix<double>;
