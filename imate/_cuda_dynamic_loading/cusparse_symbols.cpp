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

#include "./cusparse_symbols.h"
#include <cusparse.h>  // CUSPARSE_VER_MAJOR
#include <cstdlib>  // NULL
#include <sstream>  // std::ostringstream
#include "./dynamic_loading.h"  // dynamic_loading


// =========================
// Initialize static members
// =========================

cusparseDestroy_type cusparseSymbols::cusparseDestroy = NULL;
cusparseCreate_type cusparseSymbols::cusparseCreate = NULL;
cusparseCreateCsr_type cusparseSymbols::cusparseCreateCsr = NULL;
cusparseCreateDnVec_type cusparseSymbols::cusparseCreateDnVec = NULL;
cusparseDestroySpMat_type cusparseSymbols::cusparseDestroySpMat = NULL;
cusparseDestroyDnVec_type cusparseSymbols::cusparseDestroyDnVec = NULL;
cusparseSpMV_bufferSize_type cusparseSymbols::cusparseSpMV_bufferSize = NULL;
cusparseSpMV_type cusparseSymbols::cusparseSpMV = NULL;


// ============
// get lib name
// ============

/// \brief Returns the name of cusparse shared library.
///

std::string cusparseSymbols::get_lib_name()
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

    // Check cusparse version
    #ifndef CUSPARSE_VER_MAJOR
        #error "CUSPARSE_VER_MAJOR is not defined."
    #endif

    // cusparse shared library base name
    std::string lib_base_name = "libcusparse";

    // Construct the lib name
    std::ostringstream oss;
    // oss << lib_base_name << "." << lib_extension;
    oss << lib_base_name << "." << lib_extension << "." \
        << CUSPARSE_VER_MAJOR;

    std::string lib_name = oss.str();

    return lib_name;
}


#ifdef __cplusplus
    extern "C" {
#endif


// ===============
// cusparse Create
// ===============

/// \brief Definition of CUDA's \c cusparseCreate function using dynamically
///        loaded cublas library.

cusparseStatus_t cusparseCreate(cusparseHandle_t* handle)
{
    if (cusparseSymbols::cusparseCreate == NULL)
    {
        std::string lib_name = cusparseSymbols::get_lib_name();
        const char* symbol_name = "cusparseCreate";

        cusparseSymbols::cusparseCreate = \
            dynamic_loading::load_symbol<cusparseCreate_type>(
                    lib_name.c_str(),
                    symbol_name);
    }

    return cusparseSymbols::cusparseCreate(handle);
}


// ================
// cusparse Destroy
// ================

/// \brief Definition of CUDA's \c cusparseDestroy function using dynamically
///        loaded cublas library.

cusparseStatus_t cusparseDestroy(cusparseHandle_t handle)
{
    if (cusparseSymbols::cusparseDestroy == NULL)
    {
        std::string lib_name = cusparseSymbols::get_lib_name();
        const char* symbol_name = "cusparseDestroy";

        cusparseSymbols::cusparseDestroy  = \
            dynamic_loading::load_symbol<cusparseDestroy_type>(
                    lib_name.c_str(),
                    symbol_name);
    }

    return cusparseSymbols::cusparseDestroy(handle);
}


// =================
// cusparseCreateCsr
// =================

/// \brief Definition of CUDA's \c cusparseCreateCsr function using dynamically
///        loaded cublas library.

cusparseStatus_t cusparseCreateCsr(
        cusparseSpMatDescr_t* spMatDescr,
        int64_t rows,
        int64_t cols,
        int64_t nnz,
        void* csrRowOffsets,
        void* csrColInd,
        void* csrValues,
        cusparseIndexType_t csrRowOffsetsType,
        cusparseIndexType_t csrColIndType,
        cusparseIndexBase_t idxBase,
        cudaDataType valueType)
{
    if (cusparseSymbols::cusparseCreateCsr == NULL)
    {
        std::string lib_name = cusparseSymbols::get_lib_name();
        const char* symbol_name = "cusparseCreateCsr";

        cusparseSymbols::cusparseCreateCsr = \
            dynamic_loading::load_symbol<cusparseCreateCsr_type>(
                    lib_name.c_str(),
                    symbol_name);
    }

    return cusparseSymbols::cusparseCreateCsr(
            spMatDescr, rows, cols, nnz, csrRowOffsets, csrColInd, csrValues,
            csrRowOffsetsType, csrColIndType, idxBase, valueType);
}


// ===================
// cusparseCreateDnVec
// ===================

/// \brief Definition of CUDA's \c cusparseCreateDnVec function using
///        dynamically loaded cublas library.

cusparseStatus_t cusparseCreateDnVec(
        cusparseDnVecDescr_t* dnVecDescr,
        int64_t size,
        void* values,
        cudaDataType valueType)
{
    if (cusparseSymbols::cusparseCreateDnVec == NULL)
    {
        std::string lib_name = cusparseSymbols::get_lib_name();
        const char* symbol_name = "cusparseCreateDnVec";

        cusparseSymbols::cusparseCreateDnVec = \
            dynamic_loading::load_symbol<cusparseCreateDnVec_type>(
                    lib_name.c_str(),
                    symbol_name);
    }

    return cusparseSymbols::cusparseCreateDnVec(
        dnVecDescr, size, values, valueType);
}


// ====================
// cusparseDestroySpMat
// ====================

/// \brief Definition of CUDA's \c cusparseDestroySpMat function using
///        dynamically loaded cublas library.

cusparseStatus_t cusparseDestroySpMat(
        cusparseSpMatDescr_t spMatDescr)
{
    if (cusparseSymbols::cusparseDestroySpMat == NULL)
    {
        std::string lib_name = cusparseSymbols::get_lib_name();
        const char* symbol_name = "cusparseDestroySpMat";

        cusparseSymbols::cusparseDestroySpMat = \
            dynamic_loading::load_symbol<cusparseDestroySpMat_type>(
                    lib_name.c_str(),
                    symbol_name);
    }

    return cusparseSymbols::cusparseDestroySpMat(spMatDescr);
}


// ====================
// cusparseDestroyDnVec
// ====================

/// \brief Definition of CUDA's \c cusparseDestroyDnVec function using
///        dynamically loaded cublas library.

cusparseStatus_t cusparseDestroyDnVec(
        cusparseDnVecDescr_t dnVecDescr)
{
    if (cusparseSymbols::cusparseDestroyDnVec == NULL)
    {
        std::string lib_name = cusparseSymbols::get_lib_name();
        const char* symbol_name = "cusparseDestroyDnVec";

        cusparseSymbols::cusparseDestroyDnVec = \
            dynamic_loading::load_symbol<cusparseDestroyDnVec_type>(
                    lib_name.c_str(),
                    symbol_name);
    }

    return cusparseSymbols::cusparseDestroyDnVec(dnVecDescr);
}


// =======================
// cusparseSpMV_bufferSize
// =======================

/// \brief Definition of CUDA's \c cusparseSmMV_bufferSize function using
///        dynamically loaded cublas library.

cusparseStatus_t cusparseSpMV_bufferSize(
        cusparseHandle_t handle,
        cusparseOperation_t opA,
        const void* alpha,
        cusparseSpMatDescr_t matA,
        cusparseDnVecDescr_t vecX,
        const void* beta,
        cusparseDnVecDescr_t vecY,
        cudaDataType computeType,
        cusparseSpMVAlg_t alg,
        size_t* bufferSize)
{
    if (cusparseSymbols::cusparseSpMV_bufferSize == NULL)
    {
        std::string lib_name = cusparseSymbols::get_lib_name();
        const char* symbol_name = "cusparseSpMV_bufferSize";

        cusparseSymbols::cusparseSpMV_bufferSize = \
            dynamic_loading::load_symbol<cusparseSpMV_bufferSize_type>(
                    lib_name.c_str(),
                    symbol_name);
    }

    return cusparseSymbols::cusparseSpMV_bufferSize(
            handle, opA, alpha, matA, vecX, beta, vecY, computeType, alg,
            bufferSize);
}


// ============
// cusparseSpMV
// ============

/// \brief Definition of CUDA's \c cusparseSmMV function using dynamically
///        loaded cublas library.

cusparseStatus_t cusparseSpMV(
        cusparseHandle_t handle,
        cusparseOperation_t opA,
        const void* alpha,
        cusparseSpMatDescr_t matA,
        cusparseDnVecDescr_t vecX,
        const void* beta,
        cusparseDnVecDescr_t vecY,
        cudaDataType computeType,
        cusparseSpMVAlg_t alg,
        void* externalBuffer)
{
    if (cusparseSymbols::cusparseSpMV == NULL)
    {
        std::string lib_name = cusparseSymbols::get_lib_name();
        const char* symbol_name = "cusparseSpMV";

        cusparseSymbols::cusparseSpMV = \
            dynamic_loading::load_symbol<cusparseSpMV_type>(
                    lib_name.c_str(),
                    symbol_name);
    }

    return cusparseSymbols::cusparseSpMV(
            handle, opA, alpha, matA, vecX, beta, vecY, computeType, alg,
            externalBuffer);
}


#ifdef __cplusplus
    }
#endif
