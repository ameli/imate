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

#include "./cuda_interface.h"
#include <cassert>  // assert
#include <iostream>  // std::cerr
#include <cstdlib>  // abort
#include <limits>  // std::numeric_limits


// =======
// alloc 1
// =======

/// \brief     Allocates memory on gpu device. This function creates a pointer
///            and returns it.
///
///
/// \param[in] array_size
///            Size of the array to be allocated.
/// \return    A pointer to the allocated 1D array on device.

template <typename ArrayType>
ArrayType* CudaInterface<ArrayType>::alloc(const LongIndexType array_size)
{
    // Check if overflowing might make array_size negative if LongIndexType is
    // a signed type. For unsigned type, we have no clue at this point.
    assert(array_size > 0);

    // Check if computing num_bytes will not overflow size_t (unsigned int)
    size_t max_index = std::numeric_limits<size_t>::max();
    if (max_index / sizeof(ArrayType) < array_size)
    {
        std::cerr << "The size of array in bytes exceeds the maximum " \
                  << "integer limit, which is: " << max_index << ". The " \
                  << "array size is: " << array_size << ", and the size of " \
                  << "data type is: " << sizeof(ArrayType) << "-bytes." \
                  << std::endl;
        abort();
    }

    ArrayType* device_array;
    size_t num_bytes = static_cast<size_t>(array_size) * sizeof(ArrayType);
    cudaError_t error = cudaMalloc(&device_array, num_bytes);
    assert(error == cudaSuccess);

    return device_array;
}


// =======
// alloc 2
// =======

/// \brief         Allocates memory on gpu device. This function uses an
///                existing given pointer.
///
/// \param[in,out] device_array
///                A pointer to the device memory to be allocated
/// \param[in]     array_size
///                Size of the array to be allocated.

template <typename ArrayType>
void CudaInterface<ArrayType>::alloc(
        ArrayType*& device_array,
        const LongIndexType array_size)
{
    // Check if overflowing might make array_size negative if LongIndexType is
    // a signed type. For unsigned type, we have no clue at this point.
    assert(array_size > 0);

    // Check if computing num_bytes will not overflow size_t (unsigned int)
    size_t max_index = std::numeric_limits<size_t>::max();
    if (max_index / sizeof(ArrayType) < array_size)
    {
        std::cerr << "The size of array in bytes exceeds the maximum " \
                  << "integer limit, which is: " << max_index << ". The " \
                  << "array size is: " << array_size << ", and the size of " \
                  << "data type is: " << sizeof(ArrayType) << "-bytes." \
                  << std::endl;
        abort();
    }

    size_t num_bytes = static_cast<size_t>(array_size) * sizeof(ArrayType);
    cudaError_t error = cudaMalloc(&device_array, num_bytes);
    assert(error == cudaSuccess);
}


// ===========
// alloc bytes
// ===========

/// \brief         Allocates memory on gpu device. This function uses an
///                existing given pointer.
///
/// \param[in,out] device_array
///                A pointer to the device memory to be allocated
/// \param[in]     num_bytes
///                Number of bytes of the array to be allocated.

template <typename ArrayType>
void CudaInterface<ArrayType>::alloc_bytes(
        void*& device_array,
        const size_t num_bytes)
{
    // Check if overflowing might make num_bytes negative if size_t is
    // a signed type. For unsigned type, we have no clue at this point.
    assert(num_bytes > 0);

    cudaError_t error = cudaMalloc(&device_array, num_bytes);
    assert(error == cudaSuccess);
}


// ==============
// copy to device
// ==============

/// \brief      Copies memory on host to device memory.
///
/// \param[in]  host_array
///             Pointer of 1D array memory on host
/// \param[in]  array_size
///             The size of array on host.
/// \param[out] device_array
///             Pointer to the destination memory on device.

template <typename ArrayType>
void CudaInterface<ArrayType>::copy_to_device(
        const ArrayType* host_array,
        const LongIndexType array_size,
        ArrayType* device_array)
{
    size_t num_bytes = static_cast<size_t>(array_size) * sizeof(ArrayType);
    cudaError_t error = cudaMemcpy(device_array, host_array, num_bytes,
                                   cudaMemcpyHostToDevice);
    assert(error == cudaSuccess);
}


// ===
// del
// ===

/// \brief         Deletes memory on gpu device if its pointer is not \c NULL,
///                then sets the pointer to \c NULL.
///
/// \param[in,out] device_array
///                A pointer to memory on device to be deleted. This pointer
///                will be set to \c NULL.

template <typename ArrayType>
void CudaInterface<ArrayType>::del(void* device_array)
{
    if (device_array != NULL)
    {
        cudaError_t error = cudaFree(device_array);
        assert(error == cudaSuccess);
        device_array = NULL;
    }
}


// ==========
// set device
// ==========

/// \brief     Sets the current device in multi-gpu applications.
///
/// \param[in] device_id
///            The id of the device to switch to. The id is a number from \c 0
///            to \c num_gpu_devices-1

template<typename ArrayType>
void CudaInterface<ArrayType>::set_device(int device_id)
{
    cudaError_t error = cudaSetDevice(device_id);
    assert(error == cudaSuccess);
}


// ==========
// get device
// ==========

/// \brief  Gets the current device in multi-gpu applications.
///
/// \return device_id
///         The id of the current device. The id is a number from \c 0 to
///         \c num_gpu_devices-1

template<typename ArrayType>
int CudaInterface<ArrayType>::get_device()
{
    int device_id = -1;
    cudaError_t error = cudaGetDevice(&device_id);
    assert(error == cudaSuccess);

    return device_id;
}


// ===============================
// Explicit template instantiation
// ===============================

template class CudaInterface<LongIndexType>;
template class CudaInterface<float>;
template class CudaInterface<double>;
