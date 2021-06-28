/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */


#ifndef _CUDA_DYNAMIC_LOADING_CUDART_TYPES_H_
#define _CUDA_DYNAMIC_LOADING_CUDART_TYPES_H_


// =======
// Headers
// =======

#include <cuda_runtime_api.h>  // cudaError_t, cudaEvent_t, cudaStream_t,
                               // cudaDeviceProp, cudaMemcpyKind


// =====
// Types
// =====

typedef cudaError_t (*cudaEventCreate_type)(cudaEvent_t* event);
typedef cudaError_t (*cudaEventDestroy_type)(cudaEvent_t event);
typedef cudaError_t (*cudaEventElapsedTime_type)(float* ms, cudaEvent_t start,
                                            cudaEvent_t end);
typedef cudaError_t (*cudaEventRecord_type)(cudaEvent_t event,
                                            cudaStream_t stream);
typedef cudaError_t (*cudaEventSynchronize_type)(cudaEvent_t event);
typedef cudaError_t (*cudaGetDevice_type)(int* device);
typedef cudaError_t (*cudaGetDeviceCount_type)(int* count);
typedef cudaError_t (*cudaGetDeviceProperties_type)(cudaDeviceProp* prop,
                                                    int device);
typedef cudaError_t (*cudaFree_type)(void* devPtr);
typedef cudaError_t (*cudaMalloc_type)(void** devPtr, size_t size);
typedef cudaError_t (*cudaMemcpy_type)(void* dst, const void* src,
                                       size_t count, cudaMemcpyKind kind);
typedef cudaError_t (*cudaSetDevice_type)(int  device);
typedef void** (*__cudaRegisterFatBinary_type)(void *fatCubin);
typedef void (*__cudaRegisterFatBinaryEnd_type)(void **fatCubinHandle);
typedef void (*__cudaUnregisterFatBinary_type)(void **fatCubinHandle);


#endif  // _CUDA_DYNAMIC_LOADING_CUDART_TYPES_H_
