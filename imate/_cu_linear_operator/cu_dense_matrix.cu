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
#include "../_definitions/definitions.h"  // USE_OPENMP
#include "../_cu_definitions/cu_types.h" // __nv_fp8_e5m2, __nv_fp8_e4m3,
                                         // __half, __nv_bfloat16

#if defined(USE_OPENMP) && (USE_OPENMP == 1)
    #include <omp.h>  // omp_set_num_threads
#endif

#include <cstddef>  // NULL
#include <cassert>  // assert
#include "../_cu_arithmetics/cu_arithmetics.h"  // cu_arithmetics
#include "../_cu_basic_algebra/cu_matrix_operations.h"  // cuMatrixOperations
#include "../_cuda_utilities/cuda_api.h"  // alloc, copy_to_device, del


// =============
// constructor 1
// =============

/// \brief Default constructor.
///

template <typename DataType>
cuDenseMatrix<DataType>::cuDenseMatrix():

    // Initializer list
    A(NULL),
    device_A(NULL),
    A_is_row_major(0)
{
}


// =============
// constructor 2
// =============

/// \brief      Constructor.
///
/// \param[in]  A_
///             1D array that represents a 2D dense array with either C (row)
///             major ordering or Fortran (column) major ordering. The major
///             ordering should de defined by \c A_is_row_major flag.
/// \param[in]  num_rows_
///             Number of rows of \c A
/// \param[in]  num_columns_
///             Number of columns of \c A
/// \param[in]  A_is_row_major_
///             Boolean, can be \c 0 or \c 1 as follows:
///             * If \c A is row major (C ordering where the last index is
///               contiguous) this value should be \c 1.
///             * If \c A is column major (Fortran ordering where the first
///               index is contiguous), this value should be set to \c 0.
/// \param[in]  A_is_symmetric_
///             Boolean. If \c A is symmetric, set this value to \c 1,
///             otherwise \c 0.
/// \param[in]  num_gpu_devices_
///             Number of GPU devices to be utilized for parallelization.

template <typename DataType>
cuDenseMatrix<DataType>::cuDenseMatrix(
        const DataType* A_,
        const LongIndexType num_rows_,
        const LongIndexType num_columns_,
        const FlagType A_is_row_major_,
        const FlagType A_is_symmetric_,
        const int num_gpu_devices_):

    // Base class constructor
    cLinearOperatorBase(num_rows_, num_columns_),
    cuLinearOperator<DataType>(num_gpu_devices_),
    cuMatrix<DataType>(A_is_row_major_),

    // Initializer list
    A(A_),
    device_A(NULL),
    A_is_row_major(A_is_row_major_)
{
    this->initialize_cublas_handle();
    this->copy_host_to_device();
}


// ==========
// destructor
// ==========

/// \brief Destructor. This function removes data from GPU devices.
///

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
            CudaAPI<DataType>::set_device(device_id);

            // Deallocate
            CudaAPI<DataType>::del(this->device_A[device_id]);
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
        #if defined(USE_OPENMP) && (USE_OPENMP == 1)
            omp_set_num_threads(this->num_gpu_devices);
        #endif

        // Create array of pointers for data on each gpu device
        this->device_A = new DataType*[this->num_gpu_devices];

        // Size of data
        size_t A_size = static_cast<size_t>(this->num_rows) * \
                        static_cast<size_t>(this->num_columns);

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

            // Allocate device memory and copy data from host
            CudaAPI<DataType>::alloc(this->device_A[thread_id], A_size);
            CudaAPI<DataType>::copy_to_device(this->A, A_size,
                                                    this->device_A[thread_id]);
        }

        // Flag to prevent reinitialization
        this->copied_host_to_device = true;
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
FlagType cuDenseMatrix<DataType>::is_identity_matrix() const
{
    FlagType matrix_is_identity = 1;
    DataType matrix_element;
    const DataType diagonal = 1.0;
    const DataType off_diagonal = 0.0;

    // Check matrix element-wise
    if (this->A_is_row_major)
    {
        // Row-major matrix
        LongIndexType column;
        LongIndexType num_checking_columns;

        #if defined(USE_OPENMP) && (USE_OPENMP == 1)
        #pragma omp parallel for \
            schedule(static) \
            if (!omp_in_parallel()) \
            default(none) \
            shared(matrix_is_identity, diagonal, off_diagonal) \
            private(column, num_checking_columns, matrix_element)
        #endif
        for (LongIndexType row=0; row < this->num_rows; ++row)
        {
            if (matrix_is_identity)
            {
                if (this->A_is_symmetric)
                {
                    // Check only half of the columns up to diagonal element
                    num_checking_columns = row + 1;
                }
                else
                {
                    num_checking_columns = this->num_columns;
                }

                for (column=0; column < num_checking_columns; ++column)
                {
                    // Get an element of the matrix
                    matrix_element = this->A[row * this->num_columns + column];

                    // Check the value of element with identity matrix
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
    else
    {
        // Column-major matrix
        LongIndexType row;
        LongIndexType num_checking_rows;

        #if defined(USE_OPENMP) && (USE_OPENMP == 1)
        #pragma omp parallel for \
            schedule(static) \
            if (!omp_in_parallel()) \
            default(none) \
            shared(matrix_is_identity, diagonal, off_diagonal) \
            private(row, num_checking_rows, matrix_element)
        #endif
        for (LongIndexType column=0; column < this-> num_columns; ++column)
        {
            if (matrix_is_identity)
            {
                if (this->A_is_symmetric)
                {
                    // Check only half of the rows up to diagonal element
                    num_checking_rows = column + 1;
                }
                else
                {
                    num_checking_rows = this->num_rows;
                }

                for (row=0; row < num_checking_rows; ++row)
                {
                    // Get an element of the matrix
                    matrix_element = this->A[column * this->num_rows + row];

                    // Check the value of element with identity matrix
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
///             This array should be on the GPU device.
/// \param[out] device_product
///             A one-dimensional output vector \f$ \boldsymbol{y} \f$ with the
///             size of the number of rows of \f$ \mathbf{A} \f$. This vector
///             will be overwritten. This array should be on the GPU device.
///
/// \sa         cuDenseMatrix::dot_plus,
///             cuDenseMatrix::transposed_dot
///             cuDenseMatrix::transposed_dot_plus

template <typename DataType>
void cuDenseMatrix<DataType>::dot(
        const DataType* device_vector,
        DataType* device_product)
{
    assert(this->copied_host_to_device);

    // Get device id
    int device_id = CudaAPI<DataType>::get_device();

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
/// \sa         cuDenseMatrix::dot,
///             cuDenseMatrix::transposed_dot
///             cuDenseMatrix::transposed_dot_plus

template <typename DataType>
void cuDenseMatrix<DataType>::dot_plus(
        const DataType* device_vector,
        const DataType alpha,
        DataType* device_product)
{
    assert(this->copied_host_to_device);

    // Get device id
    int device_id = CudaAPI<DataType>::get_device();

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

/// \brief      Transposed-matrix vector product.
///
/// \details    Performs the matrix vector product \f$ \boldsymbol{y} =
///             \mathbf{A}^{\intercal} \boldsymbol{x} \f$.
///
/// \param[in]  device_vector
///             A one-dimensional input vector \f$ \boldsymbol{x} \f$ with size
///             the of the number of columns of the matrix \f$ \mathbf{A} \f$.
///             This array should be in GPU device.
/// \param[out] device_product
///             A one-dimensional output vector \f$ \boldsymbol{y} \f$ with the
///             size of the number of rows of \f$ \mathbf{A} \f$. This vector
///             will be overwritten. This array should be on GPU device.
///
/// \sa         cuDenseMatrix::dot_plus,
///             cuDenseMatrix::dot
///             cuDenseMatrix::transposed_dot_plus

template <typename DataType>
void cuDenseMatrix<DataType>::transpose_dot(
        const DataType* device_vector,
        DataType* device_product)
{
    assert(this->copied_host_to_device);

    // Get device id
    int device_id = CudaAPI<DataType>::get_device();

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
/// \sa         cuDenseMatrix::dot_plus,
///             cuDenseMatrix::transposed_dot
///             cuDenseMatrix::dot

template <typename DataType>
void cuDenseMatrix<DataType>::transpose_dot_plus(
        const DataType* device_vector,
        const DataType alpha,
        DataType* device_product)
{
    assert(this->copied_host_to_device);

    // Get device id
    int device_id = CudaAPI<DataType>::get_device();

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

#if defined(USE_CUDA_FP8_E5M2) && (USE_CUDA_FP8_E5M2 == 1)
    template class cuDenseMatrix<__nv_fp8_e5m2>;
#endif

#if defined(USE_CUDA_FP8_E4M3) && (USE_CUDA_FP8_E4M3 == 1)
    template class cuDenseMatrix<__nv_fp8_e4m3>;
#endif

#if defined(USE_CUDA_FP16) && (USE_CUDA_FP16 == 1)
    template class cuDenseMatrix<__half>;
#endif

#if defined(USE_CUDA_BF16) && (USE_CUDA_BF16 == 1)
    template class cuDenseMatrix<__nv_bfloat16>;
#endif

#if defined(USE_CUDA_FP32) && (USE_CUDA_FP32 == 1)
    template class cuDenseMatrix<float>;
#endif

#if defined(USE_CUDA_FP64) && (USE_CUDA_FP64 == 1)
    template class cuDenseMatrix<double>;
#endif
