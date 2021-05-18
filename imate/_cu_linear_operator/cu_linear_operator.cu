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

#include "./cu_linear_operator.h"
#include <cstddef>  // NULL
#include <cassert>  // assert
#include <cstdlib>  // abort
#include <iostream>


// =============
// constructor 1
// =============

template <typename DataType>
cuLinearOperator<DataType>::cuLinearOperator():
    copied_host_to_device(false),
    cublas_handle(NULL),
    cusparse_handle(NULL)
{
    // Check any gpu device exists
    this->query_gpu_devices();

    // Regardless of using dense (cublas) or sparse (cusparse) matrices, the
    // cublas handle should be initialized, since it is needed for the methods
    // in cuVectorOperations
    this->initialize_cublas_handle();
}


// =============
// constructor 2
// =============

/// \brief  Constructor with setting \c num_rows and \c num_columns.
///
/// \note   For the classed that are virtually derived (virtual inheritance)
///         from this class, this constructor will never be called. Rather, the
///         default constructor is called by the most derived class. Thus, set
///         the member data directly instead of below.

template <typename DataType>
cuLinearOperator<DataType>::cuLinearOperator(
        const LongIndexType num_rows_,
        const LongIndexType num_columns_):

    // Base class constructor
    cLinearOperator<DataType>(num_rows_, num_columns_),

    // Initializer list
    copied_host_to_device(false),
    cublas_handle(NULL),
    cusparse_handle(NULL)
{
    // Check any gpu device exists
    this->query_gpu_devices();

    // Regardless of using dense (cublas) or sparse (cusparse) matrices, the
    // cublas handle should be initialized, since it is needed for the methods
    // in cuVectorOperations
    this->initialize_cublas_handle();
}


// ==========
// destructor
// ==========

template <typename DataType>
cuLinearOperator<DataType>::~cuLinearOperator()
{
    if (this->cublas_handle != NULL)
    {
        cublasStatus_t status = cublasDestroy(this->cublas_handle);
        assert(status == CUBLAS_STATUS_SUCCESS);
        this->cublas_handle = NULL;
    }

    if (this->cusparse_handle != NULL)
    {
        cusparseStatus_t status = cusparseDestroy(this->cusparse_handle);
        assert(status == CUSPARSE_STATUS_SUCCESS);
        this->cusparse_handle = NULL;
    }
}


// =================
// get cublas handle
// =================

/// \brief   This function returns a reference to the \c cublasHandle_t
///          object. The object will be created, if it is not created already.
///
/// \details The \c cublasHandle is needed for the client code (slq method) for
///          vector operations on GPU. However, in this class, the
///          \c cublasHandle_t might not be needed by it self if the derived
///          class is a sparse matrix, becase the sparse matrix needs only
///          \c cusparseHandle_t. In case if the \c cublasHandle_t is not
///          created, it will be created for the purpose of the client codes.
///
/// \return  A void pointer to the cublasHandle_t instance.

template <typename DataType>
cublasHandle_t cuLinearOperator<DataType>::get_cublas_handle() const
{
    return this->cublas_handle;
}


// ========================
// initialize cublas handle
// ========================

/// \brief Creates a \c cublasHandle_t object, if not created already.
///

template <typename DataType>
void cuLinearOperator<DataType>::initialize_cublas_handle()
{
    if (this->cublas_handle == NULL)
    {
        cublasStatus_t status = cublasCreate(&this->cublas_handle);
        assert(status == CUBLAS_STATUS_SUCCESS);
    }
}


// ==========================
// initialize cusparse handle
// ==========================

/// \brief Creates a \c cusparseHandle_t object, if not created already.
///

template <typename DataType>
void cuLinearOperator<DataType>::initialize_cusparse_handle()
{
    if (this->cusparse_handle == NULL)
    {
        cusparseStatus_t status = cusparseCreate(&this->cusparse_handle);
        assert(status == CUSPARSE_STATUS_SUCCESS);
    }
}


// =================
// query gpu devices
// =================

/// \brief Before any numerical computation, this method chechs if any gpu
///        device is available on the machine, or notifies the user if nothing
///        was found.
///
/// \return Number of gpu available devices.

template <typename DataType>
int cuLinearOperator<DataType>::query_gpu_devices() const
{
    int device_count = 0;
    cudaError_t error = cudaGetDeviceCount(&device_count);

    // Error code 38 means no cuda-capable device was detected.
    if (error != cudaSuccess)
    {
        std::cerr << "ERROR: No cuda-capable GPU device was detected on " \
                  << "this machine. If a cuda-capable GPU device exists, " \
                  << "install its cuda driver. Alternatively, set " \
                  << "'gpu=False' to use cpu instead." \
                  << std::endl;
        abort();
    }

    return device_count;
}


// ===============================
// Explicit template instantiation
// ===============================

template class cuLinearOperator<float>;
template class cuLinearOperator<double>;
