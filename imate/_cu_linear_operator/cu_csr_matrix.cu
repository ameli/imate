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
#include "../_definitions/definitions.h"  // USE_OPENMP
#include "../_cu_definitions/cu_types.h" // __nv_fp8_e5m2, __nv_fp8_e4m3,
                                         // __half, __nv_bfloat16

#if defined(USE_OPENMP) && (USE_OPENMP == 1)
    #include <omp.h>  // omp_set_num_threads
#endif

#include <cstddef>  // NULL
#include <cassert>  // assert
#include "../_cu_basic_algebra/cu_matrix_operations.h"  // cuMatrixOperations
#include "../_cu_basic_algebra/cusparse_api.h"  // cusparse_api
#include "../_cuda_utilities/cuda_api.h"  // CudaAPI
#include "../_cu_arithmetics/cu_arithmetics.h"  // cu_arithmetics


// =============
// constructor 1
// =============

/// \brief Default constructor.
///

template <typename DataType>
cuCSRMatrix<DataType>::cuCSRMatrix():
    A_data(NULL),
    A_indices(NULL),
    A_index_pointer(NULL),
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
///             Number of rows of \c A
/// \param[in]  num_columns_
///             Number of columns of \c A
/// \param[in]  A_is_symmetric_
///             Boolean. If \c A is symmetric, set this value to \c 1,
///             otherwise \c 0.
/// \param[in]  num_gpu_devices_
///             Number of GPU devices to be utilzied for parallel processing.

template <typename DataType>
cuCSRMatrix<DataType>::cuCSRMatrix(
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
    cuMatrix<DataType>(A_is_symmetric_),

    // Initializer list
    A_data(A_data_),
    A_indices(A_indices_),
    A_index_pointer(A_index_pointer_),
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

/// \brief Destructor.
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
            CudaAPI<DataType>::set_device(device_id);

            // Deallocate
            CudaAPI<DataType>::del(this->device_A_data[device_id]);
            CudaAPI<LongIndexType>::del(
                    this->device_A_indices[device_id]);
            CudaAPI<LongIndexType>::del(
                    this->device_A_index_pointer[device_id]);
            CudaAPI<LongIndexType>::del(this->device_buffer[device_id]);
            cusparse_api::destroy_cusparse_matrix(
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
        #if defined(USE_OPENMP) && (USE_OPENMP == 1)
            omp_set_num_threads(this->num_gpu_devices);
        #endif

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

        #if defined(USE_OPENMP) && (USE_OPENMP == 1)
        #pragma omp parallel
        #endif
        {
            // Switch to a device with the same device id as the cpu thread id
            unsigned int thread_id;
            #if defined(USE_OPENMP) && (USE_OPENMP == 1)
                thread_id = omp_get_thread_num();
            #else
                thread_id = 0;
            #endif

            CudaAPI<DataType>::set_device(thread_id);

            // A_data
            CudaAPI<DataType>::alloc(this->device_A_data[thread_id],
                                           A_data_size);
            CudaAPI<DataType>::copy_to_device(
                    this->A_data, A_data_size, this->device_A_data[thread_id]);

            // A_indices
            CudaAPI<LongIndexType>::alloc(
                    this->device_A_indices[thread_id], A_indices_size);
            CudaAPI<LongIndexType>::copy_to_device(
                    this->A_indices, A_indices_size,
                    this->device_A_indices[thread_id]);

            // A_index_pointer
            CudaAPI<LongIndexType>::alloc(
                    this->device_A_index_pointer[thread_id],
                    A_index_pointer_size);
            CudaAPI<LongIndexType>::copy_to_device(
                    this->A_index_pointer, A_index_pointer_size,
                    this->device_A_index_pointer[thread_id]);

            // Create cusparse matrix
            cusparse_api::create_cusparse_csr_matrix(
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
///          first call of this function since buffer size is initialized to
///          zero in constructor. But for the next calls it might not be
///          reallocated if the buffer size is the same.
///
/// \param[in] device_id
///            The ID of the GPU device, from \c 0 to \c num_gpu_devices-1.
/// \param[in] cusparse_operation
///            The CuSparfse operation, which can be
///            \c CUSPARSE_OPERATION_NON_TRANSPOSE or
///            \c CUSPARSE_OPERATION_TRANSPOSE.
/// \param[in] alpha
///            Scalar. The parameter \f$ \alpha \f$ in matrix-vector
///            multiplication.
/// \param[in] beta
///            Scalar. The parameter \f$ \beta \f$ in matrix-vector
///            multiplication.
/// \param[in] cusparse_input_vector
///            Input vector in the matrix-vector multiplication.
/// \param[in] cusparse_output_vector
///            Output vector in the matrix-vector multiplication.
/// \param[in] algorithm
///            CuSparse algorithm for sparse matrix-vector product. Possible
///            values can be \c CUSPARSE_SPMV_ALG_DEFAULT,
///            \c CUSPARSE_SPMV_CSR_ALG1, \c CUSPARSE_SPMV_CSR_ALG2, etc.

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
    cusparse_api::cusparse_matrix_buffer_size(
            this->cusparse_handle[device_id], cusparse_operation, alpha,
            this->cusparse_matrix_A[device_id], cusparse_input_vector, beta,
            cusparse_output_vector, algorithm, &required_buffer_size);

    if (this->device_buffer_num_bytes[device_id] != required_buffer_size)
    {
        // Update the buffer size
        this->device_buffer_num_bytes[device_id] = required_buffer_size;

        // Delete buffer if it was allocated previously
        CudaAPI<DataType>::del(this->device_buffer[device_id]);

        // Allocate (or reallocate) buffer on device.
        CudaAPI<DataType>::alloc_bytes(
                this->device_buffer[device_id],
                this->device_buffer_num_bytes[device_id]);
    }
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
FlagType cuCSRMatrix<DataType>::is_identity_matrix() const
{
    FlagType matrix_is_identity = 1;
    LongIndexType index_pointer;
    LongIndexType column;
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
        private(index_pointer, column, matrix_element)
    #endif
    for (LongIndexType row=0; row < this->num_rows; ++row)
    {
        if (matrix_is_identity)
        {
            for (index_pointer=this->A_index_pointer[row];
                 index_pointer < this->A_index_pointer[row+1];
                 ++index_pointer)
            {
                column = this->A_indices[index_pointer];

                if (!((this->A_is_symmetric) && (column >= row)))
                {
                    matrix_element = this->A_data[index_pointer];

                    if (((row == column) && \
                         (!cu_arithmetics::is_equal(matrix_element,
                                                    diagonal))) || \
                        ((row != column) && \
                         (!cu_arithmetics::is_equal(matrix_element,
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

/// \brief   Returns the number of non-zero elements of the sparse matrix.
///
/// \details The nnz of a CSR matrix can be obtained from the last element of
///          \c A_index_pointer. The size of array \c A_index_pointer is one
///          plus the number of rows of the matrix.
///
/// \return  The nnz of the matrix.

template <typename DataType>
LongIndexType cuCSRMatrix<DataType>::get_nnz() const
{
    return this->A_index_pointer[this->num_rows];
}


// ===
// dot
// ===

/// \brief      Matrix vector product.
///
/// \details    Performs the matrix vector product \f$ \boldsymbol{y} =
///             \mathbf{A} \boldsymbol{x} \f$.
///
/// \param[in]  device_vector
///             A one-dimensional input vector \f$ \boldsymbol{x} \f$ with size
///             the of the number of columns of the matrix \f$ \mathbf{A} \f$.
///             This array should be on GPU device.
/// \param[out] device_product
///             A one-dimensional output vector \f$ \boldsymbol{y} \f$ with the
///             size of the number of rows of \f$ \mathbf{A} \f$. This vector
///             will be overwritten. This array should be on GPU device.
///
/// \sa         cuCSRMatrix::dot_plus,
///             cuCSRMatrix::transposed_dot
///             cuCSRMatrix::transposed_dot_plus

template <typename DataType>
void cuCSRMatrix<DataType>::dot(
        const DataType* device_vector,
        DataType* device_product)
{
    assert(this->copied_host_to_device);

    // Create cusparse vector for the input vector
    cusparseDnVecDescr_t cusparse_input_vector;
    cusparse_api::create_cusparse_vector(
            cusparse_input_vector, this->num_columns,
            const_cast<DataType*>(device_vector));

    // Create cusparse vector for the output vector
    cusparseDnVecDescr_t cusparse_output_vector;
    cusparse_api::create_cusparse_vector(
            cusparse_output_vector, this->num_rows, device_product);

    // Matrix vector settings
    DataType alpha = cu_arithmetics::cast<float, DataType>(1.0f);
    DataType beta = cu_arithmetics::cast<float, DataType>(0.0f);
    cusparseOperation_t cusparse_operation = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseSpMVAlg_t algorithm = CUSPARSE_SPMV_ALG_DEFAULT;

    // Get device id
    int device_id = CudaAPI<DataType>::get_device();

    // Allocate device buffer (or reallocation if needed)
    this->allocate_buffer(device_id, cusparse_operation, alpha, beta,
                          cusparse_input_vector, cusparse_output_vector,
                          algorithm);

    // Matrix vector multiplication
    cusparse_api::cusparse_matvec(
            this->cusparse_handle[device_id], cusparse_operation, alpha,
            this->cusparse_matrix_A[device_id], cusparse_input_vector, beta,
            cusparse_output_vector, algorithm, this->device_buffer[device_id]);

    // Destroy cusparse vectors
    cusparse_api::destroy_cusparse_vector(cusparse_input_vector);
    cusparse_api::destroy_cusparse_vector(cusparse_output_vector);
}


// ========
// dot plus
// ========

/// \brief      Matrix vector product written in place.
///
/// \details    Performs the matrix vector product \f$ \boldsymbol{y} =
///             \boldsymbol{y} + \alpha \mathbf{A} \boldsymbol{x} \f$.
///
/// \param[in]  device_vector
///             A one-dimensional input vector \f$ \boldsymbol{x} \f$ with size
///             the of the number of columns of the matrix \f$ \mathbf{A} \f$.
///             This array should be on GPU device.
/// \param[in]  alpha
///             A scalar.
/// \param[out] device_product
///             A one-dimensional output vector \f$ \boldsymbol{y} \f$ with the
///             size of the number of rows of \f$ \mathbf{A} \f$. This array
///             should be on GPU device.
///
/// \sa         cuCSRMatrix::dot,
///             cuCSRMatrix::transposed_dot
///             cuCSRMatrix::transposed_dot_plus

template <typename DataType>
void cuCSRMatrix<DataType>::dot_plus(
        const DataType* device_vector,
        const DataType alpha,
        DataType* device_product)
{
    assert(this->copied_host_to_device);

    // Create cusparse vector for the input vector
    cusparseDnVecDescr_t cusparse_input_vector;
    cusparse_api::create_cusparse_vector(
            cusparse_input_vector, this->num_columns,
            const_cast<DataType*>(device_vector));

    // Create cusparse vector for the output vector
    cusparseDnVecDescr_t cusparse_output_vector;
    cusparse_api::create_cusparse_vector(
            cusparse_output_vector, this->num_rows, device_product);

    // Matrix vector settings
    DataType beta = cu_arithmetics::cast<float, DataType>(1.0f);
    cusparseOperation_t cusparse_operation = CUSPARSE_OPERATION_NON_TRANSPOSE;
    cusparseSpMVAlg_t algorithm = CUSPARSE_SPMV_ALG_DEFAULT;

    // Get device id
    int device_id = CudaAPI<DataType>::get_device();

    // Allocate device buffer (or reallocation if needed)
    this->allocate_buffer(device_id, cusparse_operation, alpha, beta,
                          cusparse_input_vector, cusparse_output_vector,
                          algorithm);

    // Matrix vector multiplication
    cusparse_api::cusparse_matvec(
            this->cusparse_handle[device_id], cusparse_operation, alpha,
            this->cusparse_matrix_A[device_id], cusparse_input_vector, beta,
            cusparse_output_vector, algorithm, this->device_buffer[device_id]);

    // Destroy cusparse vectors
    cusparse_api::destroy_cusparse_vector(cusparse_input_vector);
    cusparse_api::destroy_cusparse_vector(cusparse_output_vector);
}


// =============
// transpose dot
// =============

/// \brief      Transposed-matrix vector product.
///
/// \details    Performs the matrix vector product \f$ \boldsymbol{y} =
///             \mathbf{A}^{\intercal} \boldsymbol{x} \f$.
///
/// \param[in]  device_vector
///             A one-dimensional input vector \f$ \boldsymbol{x} \f$ with size
///             the of the number of columns of the matrix \f$ \mathbf{A} \f$.
///             This array should be on GPU device.
/// \param[out] device_product
///             A one-dimensional output vector \f$ \boldsymbol{y} \f$ with the
///             size of the number of rows of \f$ \mathbf{A} \f$. This vector
///             will be overwritten. This array should be on GPU device.
///
/// \sa         cuCSRMatrix::dot_plus,
///             cuCSRMatrix::dot
///             cuCSRMatrix::transposed_dot_plus

template <typename DataType>
void cuCSRMatrix<DataType>::transpose_dot(
        const DataType* device_vector,
        DataType* device_product)
{
    assert(this->copied_host_to_device);

    // Create cusparse vector for the input vector
    cusparseDnVecDescr_t cusparse_input_vector;
    cusparse_api::create_cusparse_vector(
            cusparse_input_vector, this->num_columns,
            const_cast<DataType*>(device_vector));

    // Create cusparse vector for the output vector
    cusparseDnVecDescr_t cusparse_output_vector;
    cusparse_api::create_cusparse_vector(
            cusparse_output_vector, this->num_rows, device_product);

    // Matrix vector settings
    DataType alpha = cu_arithmetics::cast<float, DataType>(1.0f);
    DataType beta = cu_arithmetics::cast<float, DataType>(0.0f);
    cusparseOperation_t cusparse_operation = CUSPARSE_OPERATION_TRANSPOSE;
    cusparseSpMVAlg_t algorithm = CUSPARSE_SPMV_ALG_DEFAULT;

    // Get device id
    int device_id = CudaAPI<DataType>::get_device();

    // Allocate device buffer (or reallocation if needed)
    this->allocate_buffer(device_id, cusparse_operation, alpha, beta,
                          cusparse_input_vector, cusparse_output_vector,
                          algorithm);

    // Matrix vector multiplication
    cusparse_api::cusparse_matvec(
            this->cusparse_handle[device_id], cusparse_operation, alpha,
            this->cusparse_matrix_A[device_id], cusparse_input_vector, beta,
            cusparse_output_vector, algorithm, this->device_buffer[device_id]);

    // Destroy cusparse vectors
    cusparse_api::destroy_cusparse_vector(cusparse_input_vector);
    cusparse_api::destroy_cusparse_vector(cusparse_output_vector);
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
/// \param[in]  device_vector
///             A one-dimensional input vector \f$ \boldsymbol{x} \f$ with size
///             the of the number of columns of the matrix \f$ \mathbf{A} \f$.
///             This array should be on GPU device.
/// \param[in]  alpha
///             A scalar.
/// \param[out] device_product
///             A one-dimensional output vector \f$ \boldsymbol{y} \f$ with the
///             size of the number of rows of \f$ \mathbf{A} \f$. This array
///             should be on GPU device.
///
/// \sa         cuCSRMatrix::dot_plus,
///             cuCSRMatrix::transposed_dot
///             cuCSRMatrix::dot

template <typename DataType>
void cuCSRMatrix<DataType>::transpose_dot_plus(
        const DataType* device_vector,
        const DataType alpha,
        DataType* device_product)
{
    assert(this->copied_host_to_device);

    // Create cusparse vector for the input vector
    cusparseDnVecDescr_t cusparse_input_vector;
    cusparse_api::create_cusparse_vector(
            cusparse_input_vector, this->num_columns,
            const_cast<DataType*>(device_vector));

    // Create cusparse vector for the output vector
    cusparseDnVecDescr_t cusparse_output_vector;
    cusparse_api::create_cusparse_vector(
            cusparse_output_vector, this->num_rows, device_product);

    // Matrix vector settings
    DataType beta = cu_arithmetics::cast<float, DataType>(1.0f);
    cusparseOperation_t cusparse_operation = CUSPARSE_OPERATION_TRANSPOSE;
    cusparseSpMVAlg_t algorithm = CUSPARSE_SPMV_ALG_DEFAULT;

    // Get device id
    int device_id = CudaAPI<DataType>::get_device();

    // Allocate device buffer (or reallocation if needed)
    this->allocate_buffer(device_id, cusparse_operation, alpha, beta,
                          cusparse_input_vector, cusparse_output_vector,
                          algorithm);

    // Matrix vector multiplication
    cusparse_api::cusparse_matvec(
            this->cusparse_handle[device_id], cusparse_operation, alpha,
            this->cusparse_matrix_A[device_id], cusparse_input_vector, beta,
            cusparse_output_vector, algorithm, this->device_buffer[device_id]);

    // Destroy cusparse vectors
    cusparse_api::destroy_cusparse_vector(cusparse_input_vector);
    cusparse_api::destroy_cusparse_vector(cusparse_output_vector);
}


// ===============================
// Explicit template instantiation
// ===============================

#if defined(USE_CUDA_FP8_E5M2) && (USE_CUDA_FP8_E5M2 == 1)
    template class cuCSRMatrix<__nv_fp8_e5m2>;
#endif

#if defined(USE_CUDA_FP8_E4M3) && (USE_CUDA_FP8_E4M3 == 1)
    template class cuCSRMatrix<__nv_fp8_e4m3>;
#endif

#if defined(USE_CUDA_FP16) && (USE_CUDA_FP16 == 1)
    template class cuCSRMatrix<__half>;
#endif

#if defined(USE_CUDA_BF16) && (USE_CUDA_BF16 == 1)
    template class cuCSRMatrix<__nv_bfloat16>;
#endif

#if defined(USE_CUDA_FP32) && (USE_CUDA_FP32 == 1)
    template class cuCSRMatrix<float>;
#endif

#if defined(USE_CUDA_FP64) && (USE_CUDA_FP64 == 1)
    template class cuCSRMatrix<double>;
#endif
