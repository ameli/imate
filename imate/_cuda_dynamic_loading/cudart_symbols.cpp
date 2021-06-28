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


#include "./cudart_symbols.h"
#include <cuda_runtime_api.h>  // cudaError_t, cudaEvent_t, cudaStream_t,
                               // cudaDeviceProp, cudaMemcpyKind
#include <cstdlib>  // NULL
#include <sstream>  // std::ostringstream
#include "./dynamic_loading.h"  // dynamic_loading


// =========================
// Initialize static members
// =========================

cudaEventCreate_type cudartSymbols::cudaEventCreate = NULL;
cudaEventDestroy_type cudartSymbols::cudaEventDestroy = NULL;
cudaEventElapsedTime_type cudartSymbols::cudaEventElapsedTime = NULL;
cudaEventRecord_type cudartSymbols::cudaEventRecord = NULL;
cudaEventSynchronize_type cudartSymbols::cudaEventSynchronize = NULL;
cudaGetDevice_type cudartSymbols::cudaGetDevice = NULL;
cudaGetDeviceCount_type cudartSymbols::cudaGetDeviceCount = NULL;
cudaGetDeviceProperties_type cudartSymbols::cudaGetDeviceProperties = NULL;
cudaFree_type cudartSymbols::cudaFree = NULL;
cudaMalloc_type cudartSymbols::cudaMalloc = NULL;
cudaMemcpy_type cudartSymbols::cudaMemcpy = NULL;
cudaSetDevice_type cudartSymbols::cudaSetDevice = NULL;
__cudaRegisterFatBinary_type cudartSymbols::__cudaRegisterFatBinary = NULL;
__cudaRegisterFatBinaryEnd_type cudartSymbols::__cudaRegisterFatBinaryEnd = \
    NULL;
__cudaUnregisterFatBinary_type cudartSymbols::__cudaUnregisterFatBinary = NULL;


// ============
// get lib name
// ============

/// \brief Returns the name of cudart shared library.
///

std::string cudartSymbols::get_lib_name()
{
    // Get the extension name of a shared library depending on OS
    std::string lib_extension;

    #if defined(WIN32) || defined(_WIN32) || defined(__WIN32) || \
        defined(__NT__)
        lib_extension = "lib";
    #elif __APPLE__
        lib_extension = "dylib";
    #elif __linux__
        lib_extension = "so";
    #else
        #error "Unknown compiler"
    #endif

    // Check cudart version
    // #ifndef CUDART_VERSION
    //     #error "CUDART_VERSION is not defined."
    // #endif

    // CUDART_VERSION is something like 11020, which means major version 11
    // and minor version 2. To get the major version, strip off last three
    // digits.
    // int cuda_version_major = CUDART_VERSION / 1000;

    // cudart shared library base name
    std::string lib_base_name = "libcudart";

    // Construct the lib name
    std::ostringstream oss;
    oss << lib_base_name << "." << lib_extension;
    // oss << lib_base_name << "." << lib_extension << "."
    //     << std::to_string(cuda_version_major);

    std::string lib_name = oss.str();
    return lib_name;
}


#ifdef __cplusplus
    extern "C" {
#endif


// =================
// cuda Event Create
// =================

/// \brief Definition of CUDA's \c cudaEventCreate function using dynamically
///        loaded cudart library.

cudaError_t cudaEventCreate(cudaEvent_t* event)
{
    if (cudartSymbols::cudaEventCreate == NULL)
    {
        std::string lib_name = cudartSymbols::get_lib_name();
        const char* symbol_name = "cudaEventCreate";

        cudartSymbols::cudaEventCreate = \
            dynamic_loading::load_symbol<cudaEventCreate_type>(
                lib_name.c_str(),
                symbol_name);
    }

    return cudartSymbols::cudaEventCreate(event);
}


// ==================
// cuda Event Destroy
// ==================

/// \brief Definition of CUDA's \c cudaEventDestroy function using dynamically
///        loaded cudart library.

cudaError_t cudaEventDestroy(cudaEvent_t event)
{
    if (cudartSymbols::cudaEventDestroy == NULL)
    {
        std::string lib_name = cudartSymbols::get_lib_name();
        const char* symbol_name = "cudaEventDestroy";

        cudartSymbols::cudaEventDestroy = \
            dynamic_loading::load_symbol<cudaEventDestroy_type>(
                lib_name.c_str(),
                symbol_name);
    }

    return cudartSymbols::cudaEventDestroy(event);
}


// =======================
// cuda Event Elapsed Time
// =======================

/// \brief Definition of CUDA's \c cudaEventElapsedTime function using
///        dynamically loaded cudart library.

cudaError_t cudaEventElapsedTime(
        float* ms,
        cudaEvent_t start,
        cudaEvent_t end)
{
    if (cudartSymbols::cudaEventElapsedTime == NULL)
    {
        std::string lib_name = cudartSymbols::get_lib_name();
        const char* symbol_name = "cudaEventElapsedTime";

        cudartSymbols::cudaEventElapsedTime = \
            dynamic_loading::load_symbol<cudaEventElapsedTime_type>(
                lib_name.c_str(),
                symbol_name);
    }

    return cudartSymbols::cudaEventElapsedTime(ms, start, end);
}


// =================
// cuda Event Record
// =================

/// \brief Definition of CUDA's \c cudaEventRecord function using
///        dynamically loaded cudart library.

cudaError_t cudaEventRecord(cudaEvent_t event, cudaStream_t stream)
{
    if (cudartSymbols::cudaEventRecord == NULL)
    {
        std::string lib_name = cudartSymbols::get_lib_name();
        const char* symbol_name = "cudaEventRecord";

        cudartSymbols::cudaEventRecord = \
            dynamic_loading::load_symbol<cudaEventRecord_type>(
                lib_name.c_str(),
                symbol_name);
    }

    return cudartSymbols::cudaEventRecord(event, stream);
}


// ======================
// cuda Event Synchronize
// ======================

/// \brief Definition of CUDA's \c cudaEventSynchronize function using
///        dynamically loaded cudart library.

cudaError_t cudaEventSynchronize(cudaEvent_t event)
{
    if (cudartSymbols::cudaEventSynchronize == NULL)
    {
        std::string lib_name = cudartSymbols::get_lib_name();
        const char* symbol_name = "cudaEventSynchronize";

        cudartSymbols::cudaEventSynchronize = \
            dynamic_loading::load_symbol<cudaEventSynchronize_type>(
                lib_name.c_str(),
                symbol_name);
    }

    return cudartSymbols::cudaEventSynchronize(event);
}


// ===============
// cuda Get Device
// ===============

/// \brief Definition of CUDA's \c cudaGetDevice function using dynamically
///        loaded cudart library.

cudaError_t cudaGetDevice(int* device)
{
    if (cudartSymbols::cudaGetDevice == NULL)
    {
        std::string lib_name = cudartSymbols::get_lib_name();
        const char* symbol_name = "cudaGetDevice";

        cudartSymbols::cudaGetDevice = \
            dynamic_loading::load_symbol<cudaGetDevice_type>(
                lib_name.c_str(),
                symbol_name);
    }

    return cudartSymbols::cudaGetDevice(device);
}


// =====================
// cuda Get Device Count
// =====================

/// \brief Definition of CUDA's \c cudaGetDeviceCount function using
///        dynamically loaded cudart library.

cudaError_t cudaGetDeviceCount(int* count)
{
    if (cudartSymbols::cudaGetDeviceCount == NULL)
    {
        std::string lib_name = cudartSymbols::get_lib_name();
        const char* symbol_name = "cudaGetDeviceCount";

        cudartSymbols::cudaGetDeviceCount = \
            dynamic_loading::load_symbol<cudaGetDeviceCount_type>(
                lib_name.c_str(),
                symbol_name);
    }

    return cudartSymbols::cudaGetDeviceCount(count);
}


// ==========================
// cuda Get Device Properties
// ==========================

/// \brief Definition of CUDA's \c cudaGetDeviceProperties function using
///        dynamically loaded cudart library.

cudaError_t cudaGetDeviceProperties(
        cudaDeviceProp* prop,
        int device)
{
    if (cudartSymbols::cudaGetDeviceProperties == NULL)
    {
        std::string lib_name = cudartSymbols::get_lib_name();
        const char* symbol_name = "cudaGetDeviceProperties";

        cudartSymbols::cudaGetDeviceProperties = \
            dynamic_loading::load_symbol<cudaGetDeviceProperties_type>(
                lib_name.c_str(),
                symbol_name);
    }

    return cudartSymbols::cudaGetDeviceProperties(prop, device);
}


// =========
// cuda Free
// =========

/// \brief Definition of CUDA's \c cudaFree function using dynamically loaded
///        cudart library.

cudaError_t cudaFree(void* devPtr)
{
    if (cudartSymbols::cudaFree == NULL)
    {
        std::string lib_name = cudartSymbols::get_lib_name();
        const char* symbol_name = "cudaFree";

        cudartSymbols::cudaFree = dynamic_loading::load_symbol<cudaFree_type>(
                lib_name.c_str(),
                symbol_name);
    }

    return cudartSymbols::cudaFree(devPtr);
}


// ===========
// cuda Malloc
// ===========

/// \brief Definition of CUDA's \c cudaMalloc function using dynamically loaded
///        cudart library.

cudaError_t cudaMalloc(void** devPtr, size_t size)
{
    if (cudartSymbols::cudaMalloc == NULL)
    {
        std::string lib_name = cudartSymbols::get_lib_name();
        const char* symbol_name = "cudaMalloc";

        cudartSymbols::cudaMalloc = \
            dynamic_loading::load_symbol<cudaMalloc_type>(
                lib_name.c_str(),
                symbol_name);
    }

    return cudartSymbols::cudaMalloc(devPtr, size);
}


// ===========
// cuda Memcpy
// ===========

/// \brief Definition of CUDA's \c cudaMemcpy function using dynamically loaded
///        cudart library.

cudaError_t cudaMemcpy(
        void* dst,
        const void* src,
        size_t count,
        cudaMemcpyKind kind)
{
    if (cudartSymbols::cudaMemcpy == NULL)
    {
        std::string lib_name = cudartSymbols::get_lib_name();
        const char* symbol_name = "cudaMemcpy";

        cudartSymbols::cudaMemcpy = \
            dynamic_loading::load_symbol<cudaMemcpy_type>(
                lib_name.c_str(),
                symbol_name);
    }

    return cudartSymbols::cudaMemcpy(dst, src, count, kind);
}


// ===========
// cuda Memcpy
// ===========

/// \brief Definition of CUDA's \c cudaSetDevice function using dynamically
///        loaded cudart library.

cudaError_t cudaSetDevice(int  device)
{
    if (cudartSymbols::cudaSetDevice == NULL)
    {
        std::string lib_name = cudartSymbols::get_lib_name();
        const char* symbol_name = "cudaSetDevice";

        cudartSymbols::cudaSetDevice = \
            dynamic_loading::load_symbol<cudaSetDevice_type>(
                lib_name.c_str(),
                symbol_name);
    }

    return cudartSymbols::cudaSetDevice(device);
}


// ========================
// cuda Register Fat Binary
// ========================

/// \brief Definition of CUDA's \c __cudaRegisterFatBinary function using
///        dynamically loaded cudart library.

void** __cudaRegisterFatBinary(void *fatCubin)
{
    if (cudartSymbols::__cudaRegisterFatBinary == NULL)
    {
        std::string lib_name = cudartSymbols::get_lib_name();
        const char* symbol_name = "__cudaRegisterFatBinary";

        cudartSymbols::__cudaRegisterFatBinary = \
            dynamic_loading::load_symbol<__cudaRegisterFatBinary_type>(
                lib_name.c_str(),
                symbol_name);
    }

    return cudartSymbols::__cudaRegisterFatBinary(fatCubin);
}


// ============================
// cuda Register Fat Binary End
// ============================

/// \brief Definition of CUDA's \c __cudaRegisterFatBinaryEnd function using
///        dynamically loaded cudart library.

void __cudaRegisterFatBinaryEnd(void **fatCubinHandle)
{
    if (cudartSymbols::__cudaRegisterFatBinaryEnd == NULL)
    {
        std::string lib_name = cudartSymbols::get_lib_name();
        const char* symbol_name = "__cudaRegisterFatBinaryEnd";

        cudartSymbols::__cudaRegisterFatBinaryEnd = \
            dynamic_loading::load_symbol<__cudaRegisterFatBinaryEnd_type>(
                lib_name.c_str(),
                symbol_name);
    }

    return cudartSymbols::__cudaRegisterFatBinaryEnd(fatCubinHandle);
}


// ==========================
// cuda Unregister Fat Binary
// ==========================

/// \brief Definition of CUDA's \c __cudaUnregisterFatBinary function using
///        dynamically loaded cudart library.

void __cudaUnregisterFatBinary(void **fatCubinHandle)
{
    if (cudartSymbols::__cudaUnregisterFatBinary == NULL)
    {
        std::string lib_name = cudartSymbols::get_lib_name();
        const char* symbol_name = "__cudaUnregisterFatBinary";

        cudartSymbols::__cudaUnregisterFatBinary = \
            dynamic_loading::load_symbol<__cudaUnregisterFatBinary_type>(
                lib_name.c_str(),
                symbol_name);
    }

    return cudartSymbols::__cudaUnregisterFatBinary(fatCubinHandle);
}


#ifdef __cplusplus
    }
#endif
