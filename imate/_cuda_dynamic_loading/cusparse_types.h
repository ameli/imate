/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */


#ifndef _CUDA_DYNAMIC_LOADING_CUSPARSE_TYPES_H_
#define _CUDA_DYNAMIC_LOADING_CUSPARSE_TYPES_H_

// =======
// Headers
// =======

#include <cusparse.h>  // cusparseSpMatDescr_t, cusparseDnVecDescr_t,
                       // CusparseStatus_t, CUSPARSE_STATUS_SUCCESS,
                       // cusparseCreateCsr, cusparseCreateDnVec,
                       // cusparseDestroySpMat, cusparseDestroyDnVec,
                       // CUDA_R_32F, CUDA_R_64F, CUSPARSE_INDEX_32I,
                       // CUSPARSE_INDEX_BASE_ZERO, cusparseHandle_t,
                       // cusparseSpMVAlg_t, cusparseSpMV_buffer_size

// =====
// Types
// =====

// cusparse Create
typedef cusparseStatus_t (*cusparseCreate_type)(cusparseHandle_t* handle);

// cusparse Destroy
typedef cusparseStatus_t (*cusparseDestroy_type)(cusparseHandle_t handle);

// cusparseCreateCsr
typedef cusparseStatus_t (*cusparseCreateCsr_type)(
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
        cudaDataType valueType);

// cusparseCreateDnVec
typedef cusparseStatus_t (*cusparseCreateDnVec_type)(
        cusparseDnVecDescr_t* dnVecDescr,
        int64_t size,
        void* values,
        cudaDataType valueType);

// cusparseDestroySpMat
typedef cusparseStatus_t (*cusparseDestroySpMat_type)(
        cusparseSpMatDescr_t spMatDescr);

// cusparseDestroyDnVec
typedef cusparseStatus_t (*cusparseDestroyDnVec_type)(
        cusparseDnVecDescr_t dnVecDescr);

// cusparseSpMV_buffer_size
typedef cusparseStatus_t (*cusparseSpMV_bufferSize_type)(
        cusparseHandle_t handle,
        cusparseOperation_t opA,
        const void* alpha,
        cusparseSpMatDescr_t matA,
        cusparseDnVecDescr_t vecX,
        const void* beta,
        cusparseDnVecDescr_t vecY,
        cudaDataType computeType,
        cusparseSpMVAlg_t alg,
        size_t* bufferSize);

// cusparseSpMV
typedef cusparseStatus_t (*cusparseSpMV_type)(
        cusparseHandle_t handle,
        cusparseOperation_t opA,
        const void* alpha,
        cusparseSpMatDescr_t matA,
        cusparseDnVecDescr_t vecX,
        const void* beta,
        cusparseDnVecDescr_t vecY,
        cudaDataType computeType,
        cusparseSpMVAlg_t alg,
        void* externalBuffer);


#endif  // _CUDA_DYNAMIC_LOADING_CUSPARSE_TYPES_H_
