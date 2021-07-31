/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */


#ifndef _CUDA_UTILITIES_CUDA_INTERFACE_H_
#define _CUDA_UTILITIES_CUDA_INTERFACE_H_


// =======
// Headers
// =======

#include <cuda_runtime_api.h>  // cudaError_t, cudaMalloc, cudaMemcpy,
                               // cudaSuccess, cudaFree
#include "../_definitions/types.h"  // LongIndexType


// ==========
// Cuda Tools
// ==========

/// \class CudaInterface
///
/// \brief An interface to CUDA linrary to facilitate working with CUDA, such
///        as memory allocation, copy data to and from device, etc. This class
///        contains all public static functions and serves as a namespace.

template<typename ArrayType>
class CudaInterface
{
    public:

        // alloc 1
        static ArrayType* alloc(const LongIndexType array_size);

        // alloc 2
        static void alloc(
                ArrayType*& device_array,
                const LongIndexType array_size);

        // alloc bytes
        static void alloc_bytes(
                void*& device_array,
                const size_t num_bytes);

        // copy to device
        static void copy_to_device(
                const ArrayType* host_array,
                const LongIndexType array_size,
                ArrayType* device_array);

        // del
        static void del(void* device_array);

        // set device
        static void set_device(int device_id);

        // get device
        static int get_device();
};

#endif  // _CUDA_UTILITIES_CUDA_INTERFACE_H_
