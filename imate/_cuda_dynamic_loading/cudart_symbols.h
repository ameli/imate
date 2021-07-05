/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */


#ifndef _CUDA_DYNAMIC_LOADING_CUDART_SYMBOLS_H_
#define _CUDA_DYNAMIC_LOADING_CUDART_SYMBOLS_H_


// =======
// Headers
// =======

#include <string>  // std::string
#include "./cudart_types.h"  // cudaEventCreate, cudaEventDestroy,
                             // cudaEventElapsedTime, cudaEventRecord,
                             // cudaEventSynchronize,cudaGetDevice,
                             // cudaGetDeviceCount, cudaGetDeviceProperties,
                             // cudaFree, cudaMalloc, cudaMemcpy,
                             // cudaSetDevice


// ==============
// cudart Symbols
// ==============

/// \class cudartSymbols
///
/// \brief A static container to store symbols of loaded cudart library.
/// 
/// \note      When this package is compiled with dynamic loading enabled, make
///            sure that cuda toolkit is available at run-time. For instance
///            on a linux cluster, run:
///
///                module load cuda
///
/// \sa    dynamic_loading,
///        cublasSymbols,
///        cusparseSymbols

class cudartSymbols
{
    public:
        // Methods
        static std::string get_lib_name();

        // Data
        static cudaEventCreate_type cudaEventCreate;
        static cudaEventDestroy_type cudaEventDestroy;
        static cudaEventElapsedTime_type cudaEventElapsedTime;
        static cudaEventRecord_type cudaEventRecord;
        static cudaEventSynchronize_type cudaEventSynchronize;
        static cudaGetDevice_type cudaGetDevice;
        static cudaGetDeviceCount_type cudaGetDeviceCount;
        static cudaGetDeviceProperties_type cudaGetDeviceProperties;
        static cudaFree_type cudaFree;
        static cudaMalloc_type cudaMalloc;
        static cudaMemcpy_type cudaMemcpy;
        static cudaSetDevice_type cudaSetDevice;
        static __cudaRegisterFatBinary_type __cudaRegisterFatBinary;
        static __cudaRegisterFatBinaryEnd_type __cudaRegisterFatBinaryEnd;
        static __cudaUnregisterFatBinary_type __cudaUnregisterFatBinary;
};


#endif  // _CUDA_DYNAMIC_LOADING_CUDART_SYMBOLS_H_
