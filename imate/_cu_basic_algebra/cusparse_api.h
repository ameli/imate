/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */

#ifndef _CU_BASIC_ALGEBRA_CUSPARSE_API_H_
#define _CU_BASIC_ALGEBRA_CUSPARSE_API_H_


// =======
// Headers
// =======

#include <cusparse.h>  // cusparseSpMatDescr_t, cusparseDnVecDescr_t,
                       // CusparseStatus_t, CUSPARSE_STATUS_SUCCESS,
                       // cusparseCreateCsr, cusparseCreateCsc,
                       // cusparseCreateDnVec, cusparseDestroySpMat,
                       // cusparseDestroyDnVec, CUSPARSE_INDEX_BASE_ZERO,
                       // CUDA_R_16F, CUDA_R_32F, CUDA_R_64F,
                       // CUSPARSE_INDEX_32I, CUSPARSE_INDEX_64I,
                       // cusparseHandle_t, cusparseSpMVAlg_t,
                       // cusparseSpMV_buffer_size, int32_t, int64_t
#include "../_definitions/types.h"  // LongIndexType

// Restrict qualifier
#if defined(_MSC_VER)
    #define RESTRICT __restrict
#elif defined(__INTEL_COMPILER)
    #define RESTRICT __restrict
#elif defined(__CUDA__) || defined(__GNUC__) || defined(__clang__)
    #define RESTRICT __restrict__
#else
    #define RESTRICT
#endif


// ============
// cusparse api
// ============

/// \namespace cusparse_api
///
/// \brief     A collection of templates to wrapper cusparse functions.

namespace cusparse_api
{
    // create cusparse csr matrix
    template <typename DataType, typename DataIndexType>
    void create_cusparse_csr_matrix(
        cusparseSpMatDescr_t& cusparse_matrix,
        const DataIndexType num_rows,
        const DataIndexType num_columns,
        const DataIndexType nnz,
        DataType* RESTRICT device_A_data,
        DataIndexType* RESTRICT device_A_indices,
        DataIndexType* RESTRICT device_A_index_pointer);

    // create cusparse csc matrix
    template <typename DataType, typename DataIndexType>
    void create_cusparse_csc_matrix(
        cusparseSpMatDescr_t& cusparse_matrix,
        const DataIndexType num_rows,
        const DataIndexType num_columns,
        const DataIndexType nnz,
        DataType* RESTRICT device_A_data,
        DataIndexType* RESTRICT device_A_indices,
        DataIndexType* RESTRICT device_A_index_pointer);

    // create cusparse vector
    template <typename DataType>
    void create_cusparse_vector(
        cusparseDnVecDescr_t& cusparse_vector,
        const LongIndexType vector_size,
        DataType* RESTRICT device_vector);

    // destroy cusparse matrix
    void destroy_cusparse_matrix(
        cusparseSpMatDescr_t& cusparse_matrix);

    // destroy cusparse vector
    void destroy_cusparse_vector(
        cusparseDnVecDescr_t& cusparse_vector);

    // cusparse matrix buffer size
    template <typename DataType>
    void cusparse_matrix_buffer_size(
            cusparseHandle_t cusparse_handle,
            cusparseOperation_t cusparse_operation,
            const DataType alpha,
            cusparseSpMatDescr_t cusparse_matrix,
            cusparseDnVecDescr_t cusparse_input_vector,
            const DataType beta,
            cusparseDnVecDescr_t cusparse_output_vector,
            cusparseSpMVAlg_t algorithm,
            size_t* buffer_size);

    // cusparse matvec
    template <typename DataType>
    void cusparse_matvec(
            cusparseHandle_t cusparse_handle,
            cusparseOperation_t cusparse_operation,
            const DataType alpha,
            cusparseSpMatDescr_t cusparse_matrix,
            cusparseDnVecDescr_t cusparse_input_vector,
            const DataType beta,
            cusparseDnVecDescr_t cusparse_output_vector,
            cusparseSpMVAlg_t algorithm,
            void* external_buffer);

}  // namespace cusparse_api


#endif  //  _CU_BASIC_ALGEBRA_CUSPARSE_API_H_
