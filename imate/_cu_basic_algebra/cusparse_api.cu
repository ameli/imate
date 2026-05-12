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

#include "./cusparse_api.h"
#include "../_cu_definitions/cu_types.h" // __nv_fp8_e5m2, __nv_fp8_e4m3,
                                         // __half, __nv_bfloat16
#include <cassert>  // assert
#include <stdexcept>  // std::runtime_error


// ============
// cusparse api
// ============

/// \note      The implementation in the \c cu file is wrapped inside the
///            namepsace clause. This is not necessary in general, however, it
///            is needed to avoid the old gcc compiler error (this is a gcc
///            bug) which complains "no instance of function template matches
///            the argument list const float".

namespace cusparse_api
{

    // ==========================
    // create cusparse csr matrix (__nv_fp8_e5m2, int32_t)
    // ==========================

    /// \brief      A template wrapper for \c cusparseCreateCsr for the
    ///             \c __nv_fp8_e5m2 precision data and \c int32_t index type.
    ///
    /// \param[out] cusparse_matrix
    ///             Reference to CuSparse matrix object to be created.
    /// \param[in]  num_rows
    ///             Number of rows of matrix
    /// \param[in]  num_columns
    ///             Number of columns of matrix
    /// \param[in]  nnz
    ///             Number of non-zero elements of sparse matrix
    /// \param[in]  device_A_data
    ///             Array of the data of sparse matrix. The length of this
    ///             array is \c nnz. This array resides on GPU device.
    /// \param[in]  device_A_indices
    ///             Array of columns indices of sparse matrix. The length of
    ///             this array is \c nnz. This array resides on GPU device.
    /// \param[in]  device_A_index_pointer
    ///             Array of row index pointers of sparse matrix. The length
    ///             of this array is number of rows plus one. This array
    ///             resides on GPU device.
    ///
    /// \sa         create_cusparse_csc_matrix,
    ///             destroy_cusparse_matrix,
    ///             create_cusparse_vector,
    ///             destroy_cusparse_vector

    #if defined(USE_CUDA_FP8_E5M2) && (USE_CUDA_FP8_E5M2 == 1)
    template<>
    void create_cusparse_csr_matrix<__nv_fp8_e5m2, int32_t>(
            cusparseSpMatDescr_t& cusparse_matrix,
            const int32_t num_rows,
            const int32_t num_columns,
            const int32_t nnz,
            __nv_fp8_e5m2* RESTRICT device_A_data,
            int32_t* RESTRICT device_A_indices,
            int32_t* RESTRICT device_A_index_pointer)
    { 
        // TODO
        throw std::runtime_error("Function not implemented.");
    }
    #endif


    // ==========================
    // create cusparse csr matrix (__nv_fp8_e5m2, int64_t)
    // ==========================

    /// \brief      A template wrapper for \c cusparseCreateCsr for the
    ///             \c __nv_fp8_e5m2 precision data and \c int64_t index type.
    ///
    /// \param[out] cusparse_matrix
    ///             Reference to CuSparse matrix object to be created.
    /// \param[in]  num_rows
    ///             Number of rows of matrix
    /// \param[in]  num_columns
    ///             Number of columns of matrix
    /// \param[in]  nnz
    ///             Number of non-zero elements of sparse matrix
    /// \param[in]  device_A_data
    ///             Array of the data of sparse matrix. The length of this
    ///             array is \c nnz. This array resides on GPU device.
    /// \param[in]  device_A_indices
    ///             Array of columns indices of sparse matrix. The length of
    ///             this array is \c nnz. This array resides on GPU device.
    /// \param[in]  device_A_index_pointer
    ///             Array of row index pointers of sparse matrix. The length
    ///             of this array is number of rows plus one. This array
    ///             resides on GPU device.
    ///
    /// \sa         create_cusparse_csc_matrix,
    ///             destroy_cusparse_matrix,
    ///             create_cusparse_vector,
    ///             destroy_cusparse_vector

    #if defined(USE_CUDA_FP8_E5M2) && (USE_CUDA_FP8_E5M2 == 1)
    template<>
    void create_cusparse_csr_matrix<__nv_fp8_e5m2, int64_t>(
            cusparseSpMatDescr_t& cusparse_matrix,
            const int64_t num_rows,
            const int64_t num_columns,
            const int64_t nnz,
            __nv_fp8_e5m2* RESTRICT device_A_data,
            int64_t* RESTRICT device_A_indices,
            int64_t* RESTRICT device_A_index_pointer)
    {
        // TODO
        throw std::runtime_error("Function not implemented.");
    }
    #endif


    // ==========================
    // create cusparse csr matrix (__nv_fp8_e4m3, int32_t)
    // ==========================

    /// \brief      A template wrapper for \c cusparseCreateCsr for the
    ///             \c __nv_fp8_e4m3 precision data and \c int32_t index type.
    ///
    /// \param[out] cusparse_matrix
    ///             Reference to CuSparse matrix object to be created.
    /// \param[in]  num_rows
    ///             Number of rows of matrix
    /// \param[in]  num_columns
    ///             Number of columns of matrix
    /// \param[in]  nnz
    ///             Number of non-zero elements of sparse matrix
    /// \param[in]  device_A_data
    ///             Array of the data of sparse matrix. The length of this
    ///             array is \c nnz. This array resides on GPU device.
    /// \param[in]  device_A_indices
    ///             Array of columns indices of sparse matrix. The length of
    ///             this array is \c nnz. This array resides on GPU device.
    /// \param[in]  device_A_index_pointer
    ///             Array of row index pointers of sparse matrix. The length
    ///             of this array is number of rows plus one. This array
    ///             resides on GPU device.
    ///
    /// \sa         create_cusparse_csc_matrix,
    ///             destroy_cusparse_matrix,
    ///             create_cusparse_vector,
    ///             destroy_cusparse_vector

    #if defined(USE_CUDA_FP8_E4M3) && (USE_CUDA_FP8_E4M3 == 1)
    template<>
    void create_cusparse_csr_matrix<__nv_fp8_e4m3, int32_t>(
            cusparseSpMatDescr_t& cusparse_matrix,
            const int32_t num_rows,
            const int32_t num_columns,
            const int32_t nnz,
            __nv_fp8_e4m3* RESTRICT device_A_data,
            int32_t* RESTRICT device_A_indices,
            int32_t* RESTRICT device_A_index_pointer)
    {
        // TODO
        throw std::runtime_error("Function not implemented.");
    }
    #endif


    // ==========================
    // create cusparse csr matrix (__nv_fp8_e4m3, int64_t)
    // ==========================

    /// \brief      A template wrapper for \c cusparseCreateCsr for the
    ///             \c __nv_fp8_e4m3 precision data and \c int64_t index type.
    ///
    /// \param[out] cusparse_matrix
    ///             Reference to CuSparse matrix object to be created.
    /// \param[in]  num_rows
    ///             Number of rows of matrix
    /// \param[in]  num_columns
    ///             Number of columns of matrix
    /// \param[in]  nnz
    ///             Number of non-zero elements of sparse matrix
    /// \param[in]  device_A_data
    ///             Array of the data of sparse matrix. The length of this
    ///             array is \c nnz. This array resides on GPU device.
    /// \param[in]  device_A_indices
    ///             Array of columns indices of sparse matrix. The length of
    ///             this array is \c nnz. This array resides on GPU device.
    /// \param[in]  device_A_index_pointer
    ///             Array of row index pointers of sparse matrix. The length
    ///             of this array is number of rows plus one. This array
    ///             resides on GPU device.
    ///
    /// \sa         create_cusparse_csc_matrix,
    ///             destroy_cusparse_matrix,
    ///             create_cusparse_vector,
    ///             destroy_cusparse_vector

    #if defined(USE_CUDA_FP8_E4M3) && (USE_CUDA_FP8_E4M3 == 1)
    template<>
    void create_cusparse_csr_matrix<__nv_fp8_e4m3, int64_t>(
            cusparseSpMatDescr_t& cusparse_matrix,
            const int64_t num_rows,
            const int64_t num_columns,
            const int64_t nnz,
            __nv_fp8_e4m3* RESTRICT device_A_data,
            int64_t* RESTRICT device_A_indices,
            int64_t* RESTRICT device_A_index_pointer)
    {
        // TODO
        throw std::runtime_error("Function not implemented.");
    }
    #endif


    // ==========================
    // create cusparse csr matrix (__half, int32_t)
    // ==========================

    /// \brief      A template wrapper for \c cusparseCreateCsr for the
    ///             \c __half precision data and \c int32_t index type.
    ///
    /// \param[out] cusparse_matrix
    ///             Reference to CuSparse matrix object to be created.
    /// \param[in]  num_rows
    ///             Number of rows of matrix
    /// \param[in]  num_columns
    ///             Number of columns of matrix
    /// \param[in]  nnz
    ///             Number of non-zero elements of sparse matrix
    /// \param[in]  device_A_data
    ///             Array of the data of sparse matrix. The length of this
    ///             array is \c nnz. This array resides on GPU device.
    /// \param[in]  device_A_indices
    ///             Array of columns indices of sparse matrix. The length of
    ///             this array is \c nnz. This array resides on GPU device.
    /// \param[in]  device_A_index_pointer
    ///             Array of row index pointers of sparse matrix. The length
    ///             of this array is number of rows plus one. This array
    ///             resides on GPU device.
    ///
    /// \sa         create_cusparse_csc_matrix,
    ///             destroy_cusparse_matrix,
    ///             create_cusparse_vector,
    ///             destroy_cusparse_vector

    #if defined(USE_CUDA_FP16) && (USE_CUDA_FP16 == 1)
    template<>
    void create_cusparse_csr_matrix<__half, int32_t>(
            cusparseSpMatDescr_t& cusparse_matrix,
            const int32_t num_rows,
            const int32_t num_columns,
            const int32_t nnz,
            __half* RESTRICT device_A_data,
            int32_t* RESTRICT device_A_indices,
            int32_t* RESTRICT device_A_index_pointer)
    {
        cusparseStatus_t status = cusparseCreateCsr(
                &cusparse_matrix, num_rows, num_columns, nnz,
                device_A_index_pointer, device_A_indices, device_A_data,
                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16F);

        assert(status == CUSPARSE_STATUS_SUCCESS);
    }
    #endif


    // ==========================
    // create cusparse csr matrix (__half, int64_t)
    // ==========================

    /// \brief      A template wrapper for \c cusparseCreateCsr for the
    ///             \c __half precision data and \c int64_t index type.
    ///
    /// \param[out] cusparse_matrix
    ///             Reference to CuSparse matrix object to be created.
    /// \param[in]  num_rows
    ///             Number of rows of matrix
    /// \param[in]  num_columns
    ///             Number of columns of matrix
    /// \param[in]  nnz
    ///             Number of non-zero elements of sparse matrix
    /// \param[in]  device_A_data
    ///             Array of the data of sparse matrix. The length of this
    ///             array is \c nnz. This array resides on GPU device.
    /// \param[in]  device_A_indices
    ///             Array of columns indices of sparse matrix. The length of
    ///             this array is \c nnz. This array resides on GPU device.
    /// \param[in]  device_A_index_pointer
    ///             Array of row index pointers of sparse matrix. The length
    ///             of this array is number of rows plus one. This array
    ///             resides on GPU device.
    ///
    /// \sa         create_cusparse_csc_matrix,
    ///             destroy_cusparse_matrix,
    ///             create_cusparse_vector,
    ///             destroy_cusparse_vector

    #if defined(USE_CUDA_FP16) && (USE_CUDA_FP16 == 1)
    template<>
    void create_cusparse_csr_matrix<__half, int64_t>(
            cusparseSpMatDescr_t& cusparse_matrix,
            const int64_t num_rows,
            const int64_t num_columns,
            const int64_t nnz,
            __half* RESTRICT device_A_data,
            int64_t* RESTRICT device_A_indices,
            int64_t* RESTRICT device_A_index_pointer)
    {
        cusparseStatus_t status = cusparseCreateCsr(
                &cusparse_matrix, num_rows, num_columns, nnz,
                device_A_index_pointer, device_A_indices, device_A_data,
                CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I,
                CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16F);

        assert(status == CUSPARSE_STATUS_SUCCESS);
    }
    #endif


    // ==========================
    // create cusparse csr matrix (__nv_bfloat16, int32_t)
    // ==========================

    /// \brief      A template wrapper for \c cusparseCreateCsr for the
    ///             \c __nv_bfloat16 precision data and \c int32_t index type.
    ///
    /// \param[out] cusparse_matrix
    ///             Reference to CuSparse matrix object to be created.
    /// \param[in]  num_rows
    ///             Number of rows of matrix
    /// \param[in]  num_columns
    ///             Number of columns of matrix
    /// \param[in]  nnz
    ///             Number of non-zero elements of sparse matrix
    /// \param[in]  device_A_data
    ///             Array of the data of sparse matrix. The length of this
    ///             array is \c nnz. This array resides on GPU device.
    /// \param[in]  device_A_indices
    ///             Array of columns indices of sparse matrix. The length of
    ///             this array is \c nnz. This array resides on GPU device.
    /// \param[in]  device_A_index_pointer
    ///             Array of row index pointers of sparse matrix. The length
    ///             of this array is number of rows plus one. This array
    ///             resides on GPU device.
    ///
    /// \sa         create_cusparse_csc_matrix,
    ///             destroy_cusparse_matrix,
    ///             create_cusparse_vector,
    ///             destroy_cusparse_vector

    #if defined(USE_CUDA_BF16) && (USE_CUDA_BF16 == 1)
    template<>
    void create_cusparse_csr_matrix<__nv_bfloat16, int32_t>(
            cusparseSpMatDescr_t& cusparse_matrix,
            const int32_t num_rows,
            const int32_t num_columns,
            const int32_t nnz,
            __nv_bfloat16* RESTRICT device_A_data,
            int32_t* RESTRICT device_A_indices,
            int32_t* RESTRICT device_A_index_pointer)
    {
        cusparseStatus_t status = cusparseCreateCsr(
                &cusparse_matrix, num_rows, num_columns, nnz,
                device_A_index_pointer, device_A_indices, device_A_data,
                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16F);

        assert(status == CUSPARSE_STATUS_SUCCESS);
    }
    #endif


    // ==========================
    // create cusparse csr matrix (__nv_bfloat16, int64_t)
    // ==========================

    /// \brief      A template wrapper for \c cusparseCreateCsr for the
    ///             \c __nv_bfloat16 precision data and \c int64_t index type.
    ///
    /// \param[out] cusparse_matrix
    ///             Reference to CuSparse matrix object to be created.
    /// \param[in]  num_rows
    ///             Number of rows of matrix
    /// \param[in]  num_columns
    ///             Number of columns of matrix
    /// \param[in]  nnz
    ///             Number of non-zero elements of sparse matrix
    /// \param[in]  device_A_data
    ///             Array of the data of sparse matrix. The length of this
    ///             array is \c nnz. This array resides on GPU device.
    /// \param[in]  device_A_indices
    ///             Array of columns indices of sparse matrix. The length of
    ///             this array is \c nnz. This array resides on GPU device.
    /// \param[in]  device_A_index_pointer
    ///             Array of row index pointers of sparse matrix. The length
    ///             of this array is number of rows plus one. This array
    ///             resides on GPU device.
    ///
    /// \sa         create_cusparse_csc_matrix,
    ///             destroy_cusparse_matrix,
    ///             create_cusparse_vector,
    ///             destroy_cusparse_vector

    #if defined(USE_CUDA_BF16) && (USE_CUDA_BF16 == 1)
    template<>
    void create_cusparse_csr_matrix<__nv_bfloat16, int64_t>(
            cusparseSpMatDescr_t& cusparse_matrix,
            const int64_t num_rows,
            const int64_t num_columns,
            const int64_t nnz,
            __nv_bfloat16* RESTRICT device_A_data,
            int64_t* RESTRICT device_A_indices,
            int64_t* RESTRICT device_A_index_pointer)
    {
        cusparseStatus_t status = cusparseCreateCsr(
                &cusparse_matrix, num_rows, num_columns, nnz,
                device_A_index_pointer, device_A_indices, device_A_data,
                CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I,
                CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16F);

        assert(status == CUSPARSE_STATUS_SUCCESS);
    }
    #endif


    // ==========================
    // create cusparse csr matrix (float, int32_t)
    // ==========================

    /// \brief      A template wrapper for \c cusparseCreateCsr for the
    ///             \c float precision data and \c int32_t index type.
    ///
    /// \param[out] cusparse_matrix
    ///             Reference to CuSparse matrix object to be created.
    /// \param[in]  num_rows
    ///             Number of rows of matrix
    /// \param[in]  num_columns
    ///             Number of columns of matrix
    /// \param[in]  nnz
    ///             Number of non-zero elements of sparse matrix
    /// \param[in]  device_A_data
    ///             Array of the data of sparse matrix. The length of this
    ///             array is \c nnz. This array resides on GPU device.
    /// \param[in]  device_A_indices
    ///             Array of columns indices of sparse matrix. The length of
    ///             this array is \c nnz. This array resides on GPU device.
    /// \param[in]  device_A_index_pointer
    ///             Array of row index pointers of sparse matrix. The length
    ///             of this array is number of rows plus one. This array
    ///             resides on GPU device.
    ///
    /// \sa         create_cusparse_csc_matrix,
    ///             destroy_cusparse_matrix,
    ///             create_cusparse_vector,
    ///             destroy_cusparse_vector

    #if defined(USE_CUDA_FP32) && (USE_CUDA_FP32 == 1)
    template<>
    void create_cusparse_csr_matrix<float, int32_t>(
            cusparseSpMatDescr_t& cusparse_matrix,
            const int32_t num_rows,
            const int32_t num_columns,
            const int32_t nnz,
            float* RESTRICT device_A_data,
            int32_t* RESTRICT device_A_indices,
            int32_t* RESTRICT device_A_index_pointer)
    {
        cusparseStatus_t status = cusparseCreateCsr(
                &cusparse_matrix, num_rows, num_columns, nnz,
                device_A_index_pointer, device_A_indices, device_A_data,
                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

        assert(status == CUSPARSE_STATUS_SUCCESS);
    }
    #endif


    // ==========================
    // create cusparse csr matrix (float, int64_t)
    // ==========================

    /// \brief      A template wrapper for \c cusparseCreateCsr for the
    ///             \c float precision data and \c int64_t index type.
    ///
    /// \param[out] cusparse_matrix
    ///             Reference to CuSparse matrix object to be created.
    /// \param[in]  num_rows
    ///             Number of rows of matrix
    /// \param[in]  num_columns
    ///             Number of columns of matrix
    /// \param[in]  nnz
    ///             Number of non-zero elements of sparse matrix
    /// \param[in]  device_A_data
    ///             Array of the data of sparse matrix. The length of this
    ///             array is \c nnz. This array resides on GPU device.
    /// \param[in]  device_A_indices
    ///             Array of columns indices of sparse matrix. The length of
    ///             this array is \c nnz. This array resides on GPU device.
    /// \param[in]  device_A_index_pointer
    ///             Array of row index pointers of sparse matrix. The length
    ///             of this array is number of rows plus one. This array
    ///             resides on GPU device.
    ///
    /// \sa         create_cusparse_csc_matrix,
    ///             destroy_cusparse_matrix,
    ///             create_cusparse_vector,
    ///             destroy_cusparse_vector

    #if defined(USE_CUDA_FP32) && (USE_CUDA_FP32 == 1)
    template<>
    void create_cusparse_csr_matrix<float, int64_t>(
            cusparseSpMatDescr_t& cusparse_matrix,
            const int64_t num_rows,
            const int64_t num_columns,
            const int64_t nnz,
            float* RESTRICT device_A_data,
            int64_t* RESTRICT device_A_indices,
            int64_t* RESTRICT device_A_index_pointer)
    {
        cusparseStatus_t status = cusparseCreateCsr(
                &cusparse_matrix, num_rows, num_columns, nnz,
                device_A_index_pointer, device_A_indices, device_A_data,
                CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I,
                CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

        assert(status == CUSPARSE_STATUS_SUCCESS);
    }
    #endif


    // ==========================
    // create cusparse csr matrix (double, int32_t)
    // ==========================

    /// \brief      A template wrapper for \c cusparseCreateCsr for the
    ///             \c double precision data and \c int32_t index type.
    ///
    /// \param[out] cusparse_matrix
    ///             Reference to CuSparse matrix object to be created.
    /// \param[in]  num_rows
    ///             Number of rows of matrix
    /// \param[in]  num_columns
    ///             Number of columns of matrix
    /// \param[in]  nnz
    ///             Number of non-zero elements of sparse matrix
    /// \param[in]  device_A_data
    ///             Array of the data of sparse matrix. The length of this
    ///             array is \c nnz. This array resides on GPU device.
    /// \param[in]  device_A_indices
    ///             Array of columns indices of sparse matrix. The length of
    ///             this array is \c nnz. This array resides on GPU device.
    /// \param[in]  device_A_index_pointer
    ///             Array of row index pointers of sparse matrix. The length
    ///             of this array is number of rows plus one. This array
    ///             resides on GPU device.
    ///
    /// \sa         create_cusparse_csc_matrix,
    ///             destroy_cusparse_matrix,
    ///             create_cusparse_vector,
    ///             destroy_cusparse_vector

    #if defined(USE_CUDA_FP64) && (USE_CUDA_FP64 == 1)
    template<>
    void create_cusparse_csr_matrix<double, int32_t>(
            cusparseSpMatDescr_t& cusparse_matrix,
            const int32_t num_rows,
            const int32_t num_columns,
            const int32_t nnz,
            double* RESTRICT device_A_data,
            int32_t* RESTRICT device_A_indices,
            int32_t* RESTRICT device_A_index_pointer)
    {
        cusparseStatus_t status = cusparseCreateCsr(
                &cusparse_matrix, num_rows, num_columns, nnz,
                device_A_index_pointer, device_A_indices, device_A_data,
                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

        assert(status == CUSPARSE_STATUS_SUCCESS);
    }
    #endif


    // ==========================
    // create cusparse csr matrix (double, int64_t)
    // ==========================

    /// \brief      A template wrapper for \c cusparseCreateCsr for the
    ///             \c double precision data and \c int64_t index type.
    ///
    /// \param[out] cusparse_matrix
    ///             Reference to CuSparse matrix object to be created.
    /// \param[in]  num_rows
    ///             Number of rows of matrix
    /// \param[in]  num_columns
    ///             Number of columns of matrix
    /// \param[in]  nnz
    ///             Number of non-zero elements of sparse matrix
    /// \param[in]  device_A_data
    ///             Array of the data of sparse matrix. The length of this
    ///             array is \c nnz. This array resides on GPU device.
    /// \param[in]  device_A_indices
    ///             Array of columns indices of sparse matrix. The length of
    ///             this array is \c nnz. This array resides on GPU device.
    /// \param[in]  device_A_index_pointer
    ///             Array of row index pointers of sparse matrix. The length
    ///             of this array is number of rows plus one. This array
    ///             resides on GPU device.
    ///
    /// \sa         create_cusparse_csc_matrix,
    ///             destroy_cusparse_matrix,
    ///             create_cusparse_vector,
    ///             destroy_cusparse_vector

    #if defined(USE_CUDA_FP64) && (USE_CUDA_FP64 == 1)
    template<>
    void create_cusparse_csr_matrix<double, int64_t>(
            cusparseSpMatDescr_t& cusparse_matrix,
            const int64_t num_rows,
            const int64_t num_columns,
            const int64_t nnz,
            double* RESTRICT device_A_data,
            int64_t* RESTRICT device_A_indices,
            int64_t* RESTRICT device_A_index_pointer)
    {
        cusparseStatus_t status = cusparseCreateCsr(
                &cusparse_matrix, num_rows, num_columns, nnz,
                device_A_index_pointer, device_A_indices, device_A_data,
                CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I,
                CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

        assert(status == CUSPARSE_STATUS_SUCCESS);
    }
    #endif


    // ==========================
    // create cusparse csc matrix (__nv_fp8_e5m2, int32_t)
    // ==========================

    /// \brief      A template wrapper for \c cusparseCreateCsc for the
    ///             \c __nv_fp8_e5m2 precision data and \c int32_t index type.
    ///
    /// \param[out] cusparse_matrix
    ///             Reference to CuSparse matrix object to be created.
    /// \param[in]  num_rows
    ///             Number of rows of matrix
    /// \param[in]  num_columns
    ///             Number of columns of matrix
    /// \param[in]  nnz
    ///             Number of non-zero elements of sparse matrix
    /// \param[in]  device_A_data
    ///             Array of the data of sparse matrix. The length of this
    ///             array is \c nnz. This array resides on GPU device.
    /// \param[in]  device_A_indices
    ///             Array of rows indices of sparse matrix. The length of
    ///             this array is \c nnz. This array resides on GPU device.
    /// \param[in]  device_A_index_pointer
    ///             Array of row index pointers of sparse matrix. The length
    ///             of this array is number of columns plus one. This array
    ///             resides on GPU device.
    ///
    /// \sa         create_cusparse_csr_matrix,
    ///             destroy_cusparse_matrix,
    ///             create_cusparse_vector,
    ///             destroy_cusparse_vector

    #if defined(USE_CUDA_FP8_E5M2) && (USE_CUDA_FP8_E5M2 == 1)
    template<>
    void create_cusparse_csc_matrix<__nv_fp8_e5m2, int32_t>(
            cusparseSpMatDescr_t& cusparse_matrix,
            const int32_t num_rows,
            const int32_t num_columns,
            const int32_t nnz,
            __nv_fp8_e5m2* RESTRICT device_A_data,
            int32_t* RESTRICT device_A_indices,
            int32_t* RESTRICT device_A_index_pointer)
    {
        // TODO
        throw std::runtime_error("Function not implemented.");
    }
    #endif


    // ==========================
    // create cusparse csc matrix (__nv_fp8_e5m2, int64_t)
    // ==========================

    /// \brief      A template wrapper for \c cusparseCreateCsc for the
    ///             \c __nv_fp8_e5m2 precision data and \c int64_t index type.
    ///
    /// \param[out] cusparse_matrix
    ///             Reference to CuSparse matrix object to be created.
    /// \param[in]  num_rows
    ///             Number of rows of matrix
    /// \param[in]  num_columns
    ///             Number of columns of matrix
    /// \param[in]  nnz
    ///             Number of non-zero elements of sparse matrix
    /// \param[in]  device_A_data
    ///             Array of the data of sparse matrix. The length of this
    ///             array is \c nnz. This array resides on GPU device.
    /// \param[in]  device_A_indices
    ///             Array of rows indices of sparse matrix. The length of
    ///             this array is \c nnz. This array resides on GPU device.
    /// \param[in]  device_A_index_pointer
    ///             Array of row index pointers of sparse matrix. The length
    ///             of this array is number of columns plus one. This array
    ///             resides on GPU device.
    ///
    /// \sa         create_cusparse_csr_matrix,
    ///             destroy_cusparse_matrix,
    ///             create_cusparse_vector,
    ///             destroy_cusparse_vector

    #if defined(USE_CUDA_FP8_E5M2) && (USE_CUDA_FP8_E5M2 == 1)
    template<>
    void create_cusparse_csc_matrix<__nv_fp8_e5m2, int64_t>(
            cusparseSpMatDescr_t& cusparse_matrix,
            const int64_t num_rows,
            const int64_t num_columns,
            const int64_t nnz,
            __nv_fp8_e5m2* RESTRICT device_A_data,
            int64_t* RESTRICT device_A_indices,
            int64_t* RESTRICT device_A_index_pointer)
    {
        // TODO
        throw std::runtime_error("Function not implemented.");
    }
    #endif


    // ==========================
    // create cusparse csc matrix (__nv_fp8_e4m3, int32_t)
    // ==========================

    /// \brief      A template wrapper for \c cusparseCreateCsc for the
    ///             \c __nv_fp8_e4m3 precision data and \c int32_t index type.
    ///
    /// \param[out] cusparse_matrix
    ///             Reference to CuSparse matrix object to be created.
    /// \param[in]  num_rows
    ///             Number of rows of matrix
    /// \param[in]  num_columns
    ///             Number of columns of matrix
    /// \param[in]  nnz
    ///             Number of non-zero elements of sparse matrix
    /// \param[in]  device_A_data
    ///             Array of the data of sparse matrix. The length of this
    ///             array is \c nnz. This array resides on GPU device.
    /// \param[in]  device_A_indices
    ///             Array of rows indices of sparse matrix. The length of
    ///             this array is \c nnz. This array resides on GPU device.
    /// \param[in]  device_A_index_pointer
    ///             Array of row index pointers of sparse matrix. The length
    ///             of this array is number of columns plus one. This array
    ///             resides on GPU device.
    ///
    /// \sa         create_cusparse_csr_matrix,
    ///             destroy_cusparse_matrix,
    ///             create_cusparse_vector,
    ///             destroy_cusparse_vector

    #if defined(USE_CUDA_FP8_E4M3) && (USE_CUDA_FP8_E4M3 == 1)
    template<>
    void create_cusparse_csc_matrix<__nv_fp8_e4m3, int32_t>(
            cusparseSpMatDescr_t& cusparse_matrix,
            const int32_t num_rows,
            const int32_t num_columns,
            const int32_t nnz,
            __nv_fp8_e4m3* RESTRICT device_A_data,
            int32_t* RESTRICT device_A_indices,
            int32_t* RESTRICT device_A_index_pointer)
    {
        // TODO
        throw std::runtime_error("Function not implemented.");
    }
    #endif


    // ==========================
    // create cusparse csc matrix (__nv_fp8_e4m3, int64_t)
    // ==========================

    /// \brief      A template wrapper for \c cusparseCreateCsc for the
    ///             \c __nv_fp8_e4m3 precision data and \c int64_t index type.
    ///
    /// \param[out] cusparse_matrix
    ///             Reference to CuSparse matrix object to be created.
    /// \param[in]  num_rows
    ///             Number of rows of matrix
    /// \param[in]  num_columns
    ///             Number of columns of matrix
    /// \param[in]  nnz
    ///             Number of non-zero elements of sparse matrix
    /// \param[in]  device_A_data
    ///             Array of the data of sparse matrix. The length of this
    ///             array is \c nnz. This array resides on GPU device.
    /// \param[in]  device_A_indices
    ///             Array of rows indices of sparse matrix. The length of
    ///             this array is \c nnz. This array resides on GPU device.
    /// \param[in]  device_A_index_pointer
    ///             Array of row index pointers of sparse matrix. The length
    ///             of this array is number of columns plus one. This array
    ///             resides on GPU device.
    ///
    /// \sa         create_cusparse_csr_matrix,
    ///             destroy_cusparse_matrix,
    ///             create_cusparse_vector,
    ///             destroy_cusparse_vector

    #if defined(USE_CUDA_FP8_E4M3) && (USE_CUDA_FP8_E4M3 == 1)
    template<>
    void create_cusparse_csc_matrix<__nv_fp8_e4m3, int64_t>(
            cusparseSpMatDescr_t& cusparse_matrix,
            const int64_t num_rows,
            const int64_t num_columns,
            const int64_t nnz,
            __nv_fp8_e4m3* RESTRICT device_A_data,
            int64_t* RESTRICT device_A_indices,
            int64_t* RESTRICT device_A_index_pointer)
    {
        // TODO
        throw std::runtime_error("Function not implemented.");
    }
    #endif


    // ==========================
    // create cusparse csc matrix (__half, int32_t)
    // ==========================

    /// \brief      A template wrapper for \c cusparseCreateCsc for the
    ///             \c __half precision data and \c int32_t index type.
    ///
    /// \param[out] cusparse_matrix
    ///             Reference to CuSparse matrix object to be created.
    /// \param[in]  num_rows
    ///             Number of rows of matrix
    /// \param[in]  num_columns
    ///             Number of columns of matrix
    /// \param[in]  nnz
    ///             Number of non-zero elements of sparse matrix
    /// \param[in]  device_A_data
    ///             Array of the data of sparse matrix. The length of this
    ///             array is \c nnz. This array resides on GPU device.
    /// \param[in]  device_A_indices
    ///             Array of rows indices of sparse matrix. The length of
    ///             this array is \c nnz. This array resides on GPU device.
    /// \param[in]  device_A_index_pointer
    ///             Array of row index pointers of sparse matrix. The length
    ///             of this array is number of columns plus one. This array
    ///             resides on GPU device.
    ///
    /// \sa         create_cusparse_csr_matrix,
    ///             destroy_cusparse_matrix,
    ///             create_cusparse_vector,
    ///             destroy_cusparse_vector

    #if defined(USE_CUDA_FP16) && (USE_CUDA_FP16 == 1)
    template<>
    void create_cusparse_csc_matrix<__half, int32_t>(
            cusparseSpMatDescr_t& cusparse_matrix,
            const int32_t num_rows,
            const int32_t num_columns,
            const int32_t nnz,
            __half* RESTRICT device_A_data,
            int32_t* RESTRICT device_A_indices,
            int32_t* RESTRICT device_A_index_pointer)
    {
        cusparseStatus_t status = cusparseCreateCsc(
                &cusparse_matrix, num_rows, num_columns, nnz,
                device_A_index_pointer, device_A_indices, device_A_data,
                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16F);

        assert(status == CUSPARSE_STATUS_SUCCESS);
    }
    #endif


    // ==========================
    // create cusparse csc matrix (__half, int64_t)
    // ==========================

    /// \brief      A template wrapper for \c cusparseCreateCsc for the
    ///             \c __half precision data and \c int64_t index type.
    ///
    /// \param[out] cusparse_matrix
    ///             Reference to CuSparse matrix object to be created.
    /// \param[in]  num_rows
    ///             Number of rows of matrix
    /// \param[in]  num_columns
    ///             Number of columns of matrix
    /// \param[in]  nnz
    ///             Number of non-zero elements of sparse matrix
    /// \param[in]  device_A_data
    ///             Array of the data of sparse matrix. The length of this
    ///             array is \c nnz. This array resides on GPU device.
    /// \param[in]  device_A_indices
    ///             Array of rows indices of sparse matrix. The length of
    ///             this array is \c nnz. This array resides on GPU device.
    /// \param[in]  device_A_index_pointer
    ///             Array of row index pointers of sparse matrix. The length
    ///             of this array is number of columns plus one. This array
    ///             resides on GPU device.
    ///
    /// \sa         create_cusparse_csr_matrix,
    ///             destroy_cusparse_matrix,
    ///             create_cusparse_vector,
    ///             destroy_cusparse_vector

    #if defined(USE_CUDA_FP16) && (USE_CUDA_FP16 == 1)
    template<>
    void create_cusparse_csc_matrix<__half, int64_t>(
            cusparseSpMatDescr_t& cusparse_matrix,
            const int64_t num_rows,
            const int64_t num_columns,
            const int64_t nnz,
            __half* RESTRICT device_A_data,
            int64_t* RESTRICT device_A_indices,
            int64_t* RESTRICT device_A_index_pointer)
    {
        cusparseStatus_t status = cusparseCreateCsc(
                &cusparse_matrix, num_rows, num_columns, nnz,
                device_A_index_pointer, device_A_indices, device_A_data,
                CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I,
                CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16F);

        assert(status == CUSPARSE_STATUS_SUCCESS);
    }
    #endif


    // ==========================
    // create cusparse csc matrix (__nv_bfloat16, int32_t)
    // ==========================

    /// \brief      A template wrapper for \c cusparseCreateCsc for the
    ///             \c __nv_bfloat16 precision data and \c int32_t index type.
    ///
    /// \param[out] cusparse_matrix
    ///             Reference to CuSparse matrix object to be created.
    /// \param[in]  num_rows
    ///             Number of rows of matrix
    /// \param[in]  num_columns
    ///             Number of columns of matrix
    /// \param[in]  nnz
    ///             Number of non-zero elements of sparse matrix
    /// \param[in]  device_A_data
    ///             Array of the data of sparse matrix. The length of this
    ///             array is \c nnz. This array resides on GPU device.
    /// \param[in]  device_A_indices
    ///             Array of rows indices of sparse matrix. The length of
    ///             this array is \c nnz. This array resides on GPU device.
    /// \param[in]  device_A_index_pointer
    ///             Array of row index pointers of sparse matrix. The length
    ///             of this array is number of columns plus one. This array
    ///             resides on GPU device.
    ///
    /// \sa         create_cusparse_csr_matrix,
    ///             destroy_cusparse_matrix,
    ///             create_cusparse_vector,
    ///             destroy_cusparse_vector

    #if defined(USE_CUDA_BF16) && (USE_CUDA_BF16 == 1)
    template<>
    void create_cusparse_csc_matrix<__nv_bfloat16, int32_t>(
            cusparseSpMatDescr_t& cusparse_matrix,
            const int32_t num_rows,
            const int32_t num_columns,
            const int32_t nnz,
            __nv_bfloat16* RESTRICT device_A_data,
            int32_t* RESTRICT device_A_indices,
            int32_t* RESTRICT device_A_index_pointer)
    {
        cusparseStatus_t status = cusparseCreateCsc(
                &cusparse_matrix, num_rows, num_columns, nnz,
                device_A_index_pointer, device_A_indices, device_A_data,
                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16F);

        assert(status == CUSPARSE_STATUS_SUCCESS);
    }
    #endif


    // ==========================
    // create cusparse csc matrix (__nv_bfloat16, int64_t)
    // ==========================

    /// \brief      A template wrapper for \c cusparseCreateCsc for the
    ///             \c __nv_bfloat16 precision data and \c int64_t index type.
    ///
    /// \param[out] cusparse_matrix
    ///             Reference to CuSparse matrix object to be created.
    /// \param[in]  num_rows
    ///             Number of rows of matrix
    /// \param[in]  num_columns
    ///             Number of columns of matrix
    /// \param[in]  nnz
    ///             Number of non-zero elements of sparse matrix
    /// \param[in]  device_A_data
    ///             Array of the data of sparse matrix. The length of this
    ///             array is \c nnz. This array resides on GPU device.
    /// \param[in]  device_A_indices
    ///             Array of rows indices of sparse matrix. The length of
    ///             this array is \c nnz. This array resides on GPU device.
    /// \param[in]  device_A_index_pointer
    ///             Array of row index pointers of sparse matrix. The length
    ///             of this array is number of columns plus one. This array
    ///             resides on GPU device.
    ///
    /// \sa         create_cusparse_csr_matrix,
    ///             destroy_cusparse_matrix,
    ///             create_cusparse_vector,
    ///             destroy_cusparse_vector

    #if defined(USE_CUDA_BF16) && (USE_CUDA_BF16 == 1)
    template<>
    void create_cusparse_csc_matrix<__nv_bfloat16, int64_t>(
            cusparseSpMatDescr_t& cusparse_matrix,
            const int64_t num_rows,
            const int64_t num_columns,
            const int64_t nnz,
            __nv_bfloat16* RESTRICT device_A_data,
            int64_t* RESTRICT device_A_indices,
            int64_t* RESTRICT device_A_index_pointer)
    {
        cusparseStatus_t status = cusparseCreateCsc(
                &cusparse_matrix, num_rows, num_columns, nnz,
                device_A_index_pointer, device_A_indices, device_A_data,
                CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I,
                CUSPARSE_INDEX_BASE_ZERO, CUDA_R_16F);

        assert(status == CUSPARSE_STATUS_SUCCESS);
    }
    #endif


    // ==========================
    // create cusparse csc matrix (float, int32_t)
    // ==========================

    /// \brief      A template wrapper for \c cusparseCreateCsc for the
    ///             \c float precision data and \c int32_t index type.
    ///
    /// \param[out] cusparse_matrix
    ///             Reference to CuSparse matrix object to be created.
    /// \param[in]  num_rows
    ///             Number of rows of matrix
    /// \param[in]  num_columns
    ///             Number of columns of matrix
    /// \param[in]  nnz
    ///             Number of non-zero elements of sparse matrix
    /// \param[in]  device_A_data
    ///             Array of the data of sparse matrix. The length of this
    ///             array is \c nnz. This array resides on GPU device.
    /// \param[in]  device_A_indices
    ///             Array of rows indices of sparse matrix. The length of
    ///             this array is \c nnz. This array resides on GPU device.
    /// \param[in]  device_A_index_pointer
    ///             Array of row index pointers of sparse matrix. The length
    ///             of this array is number of columns plus one. This array
    ///             resides on GPU device.
    ///
    /// \sa         create_cusparse_csr_matrix,
    ///             destroy_cusparse_matrix,
    ///             create_cusparse_vector,
    ///             destroy_cusparse_vector

    #if defined(USE_CUDA_FP32) && (USE_CUDA_FP32 == 1)
    template<>
    void create_cusparse_csc_matrix<float, int32_t>(
            cusparseSpMatDescr_t& cusparse_matrix,
            const int32_t num_rows,
            const int32_t num_columns,
            const int32_t nnz,
            float* RESTRICT device_A_data,
            int32_t* RESTRICT device_A_indices,
            int32_t* RESTRICT device_A_index_pointer)
    {
        cusparseStatus_t status = cusparseCreateCsc(
                &cusparse_matrix, num_rows, num_columns, nnz,
                device_A_index_pointer, device_A_indices, device_A_data,
                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

        assert(status == CUSPARSE_STATUS_SUCCESS);
    }
    #endif


    // ==========================
    // create cusparse csc matrix (float, int64_t)
    // ==========================

    /// \brief      A template wrapper for \c cusparseCreateCsc for the
    ///             \c float precision data and \c int64_t index type.
    ///
    /// \param[out] cusparse_matrix
    ///             Reference to CuSparse matrix object to be created.
    /// \param[in]  num_rows
    ///             Number of rows of matrix
    /// \param[in]  num_columns
    ///             Number of columns of matrix
    /// \param[in]  nnz
    ///             Number of non-zero elements of sparse matrix
    /// \param[in]  device_A_data
    ///             Array of the data of sparse matrix. The length of this
    ///             array is \c nnz. This array resides on GPU device.
    /// \param[in]  device_A_indices
    ///             Array of rows indices of sparse matrix. The length of
    ///             this array is \c nnz. This array resides on GPU device.
    /// \param[in]  device_A_index_pointer
    ///             Array of row index pointers of sparse matrix. The length
    ///             of this array is number of columns plus one. This array
    ///             resides on GPU device.
    ///
    /// \sa         create_cusparse_csr_matrix,
    ///             destroy_cusparse_matrix,
    ///             create_cusparse_vector,
    ///             destroy_cusparse_vector

    #if defined(USE_CUDA_FP32) && (USE_CUDA_FP32 == 1)
    template<>
    void create_cusparse_csc_matrix<float, int64_t>(
            cusparseSpMatDescr_t& cusparse_matrix,
            const int64_t num_rows,
            const int64_t num_columns,
            const int64_t nnz,
            float* RESTRICT device_A_data,
            int64_t* RESTRICT device_A_indices,
            int64_t* RESTRICT device_A_index_pointer)
    {
        cusparseStatus_t status = cusparseCreateCsc(
                &cusparse_matrix, num_rows, num_columns, nnz,
                device_A_index_pointer, device_A_indices, device_A_data,
                CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I,
                CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

        assert(status == CUSPARSE_STATUS_SUCCESS);
    }
    #endif


    // ==========================
    // create cusparse csc matrix (double, int32_t)
    // ==========================

    /// \brief      A template wrapper for \c cusparseCreateCsc for the
    ///             \c double precision data and \c int32_t index type.
    ///
    /// \param[out] cusparse_matrix
    ///             Reference to CuSparse matrix object to be created.
    /// \param[in]  num_rows
    ///             Number of rows of matrix
    /// \param[in]  num_columns
    ///             Number of columns of matrix
    /// \param[in]  nnz
    ///             Number of non-zero elements of sparse matrix
    /// \param[in]  device_A_data
    ///             Array of the data of sparse matrix. The length of this
    ///             array is \c nnz. This array resides on GPU device.
    /// \param[in]  device_A_indices
    ///             Array of rows indices of sparse matrix. The length of
    ///             this array is \c nnz. This array resides on GPU device.
    /// \param[in]  device_A_index_pointer
    ///             Array of row index pointers of sparse matrix. The length
    ///             of this array is number of columns plus one. This array
    ///             resides on GPU device.
    ///
    /// \sa         create_cusparse_csr_matrix,
    ///             destroy_cusparse_matrix,
    ///             create_cusparse_vector,
    ///             destroy_cusparse_vector

    #if defined(USE_CUDA_FP64) && (USE_CUDA_FP64 == 1)
    template<>
    void create_cusparse_csc_matrix<double, int32_t>(
            cusparseSpMatDescr_t& cusparse_matrix,
            const int32_t num_rows,
            const int32_t num_columns,
            const int32_t nnz,
            double* RESTRICT device_A_data,
            int32_t* RESTRICT device_A_indices,
            int32_t* RESTRICT device_A_index_pointer)
    {
        cusparseStatus_t status = cusparseCreateCsc(
                &cusparse_matrix, num_rows, num_columns, nnz,
                device_A_index_pointer, device_A_indices, device_A_data,
                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

        assert(status == CUSPARSE_STATUS_SUCCESS);
    }
    #endif


    // ==========================
    // create cusparse csc matrix (double, int64_t)
    // ==========================

    /// \brief      A template wrapper for \c cusparseCreateCsc for the
    ///             \c double precision data and \c int64_t index type.
    ///
    /// \param[out] cusparse_matrix
    ///             Reference to CuSparse matrix object to be created.
    /// \param[in]  num_rows
    ///             Number of rows of matrix
    /// \param[in]  num_columns
    ///             Number of columns of matrix
    /// \param[in]  nnz
    ///             Number of non-zero elements of sparse matrix
    /// \param[in]  device_A_data
    ///             Array of the data of sparse matrix. The length of this
    ///             array is \c nnz. This array resides on GPU device.
    /// \param[in]  device_A_indices
    ///             Array of rows indices of sparse matrix. The length of
    ///             this array is \c nnz. This array resides on GPU device.
    /// \param[in]  device_A_index_pointer
    ///             Array of row index pointers of sparse matrix. The length
    ///             of this array is number of columns plus one. This array
    ///             resides on GPU device.
    ///
    /// \sa         create_cusparse_csr_matrix,
    ///             destroy_cusparse_matrix,
    ///             create_cusparse_vector,
    ///             destroy_cusparse_vector

    #if defined(USE_CUDA_FP64) && (USE_CUDA_FP64 == 1)
    template<>
    void create_cusparse_csc_matrix<double, int64_t>(
            cusparseSpMatDescr_t& cusparse_matrix,
            const int64_t num_rows,
            const int64_t num_columns,
            const int64_t nnz,
            double* RESTRICT device_A_data,
            int64_t* RESTRICT device_A_indices,
            int64_t* RESTRICT device_A_index_pointer)
    {
        cusparseStatus_t status = cusparseCreateCsc(
                &cusparse_matrix, num_rows, num_columns, nnz,
                device_A_index_pointer, device_A_indices, device_A_data,
                CUSPARSE_INDEX_64I, CUSPARSE_INDEX_64I,
                CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

        assert(status == CUSPARSE_STATUS_SUCCESS);
    }
    #endif


    // ======================
    // create cusparse vector (__nv_fp8_e5m2)
    // ======================

    /// \brief      A template wrapper for \c cusparseCreateDnVec for the
    ///             \c __nv_fp8_e5m2 precision data.
    ///
    /// \details    Note that according to the cusparse documentation for the
    ///             function \c cusparseCreateDnVec, it is safe to use
    ///             \c const_cast to cast the input vector.
    ///
    /// \param[out] cusparse_vector
    ///             Reference to CuSparse vector object to be created.
    /// \param[in]  vector_size
    ///             Size of vector array
    /// \param[in]  device_vector
    ///             Array data of vector on GPU device.
    ///
    /// \sa         destroy_cusparse_vector,
    ///             create_cusparse_csr_matrix,
    ///             create_cusparse_csc_matrix

    #if defined(USE_CUDA_FP8_E5M2) && (USE_CUDA_FP8_E5M2 == 1)
    template<>
    void create_cusparse_vector<__nv_fp8_e5m2>(
            cusparseDnVecDescr_t& cusparse_vector,
            const LongIndexType vector_size,
            __nv_fp8_e5m2* RESTRICT device_vector)
    {
        // TODO
        throw std::runtime_error("Function not implemented.");
    }
    #endif


    // ======================
    // create cusparse vector (__nv_fp8_e4m3)
    // ======================

    /// \brief      A template wrapper for \c cusparseCreateDnVec for the
    ///             \c __nv_fp8_e4m3 precision data.
    ///
    /// \details    Note that according to the cusparse documentation for the
    ///             function \c cusparseCreateDnVec, it is safe to use
    ///             \c const_cast to cast the input vector.
    ///
    /// \param[out] cusparse_vector
    ///             Reference to CuSparse vector object to be created.
    /// \param[in]  vector_size
    ///             Size of vector array
    /// \param[in]  device_vector
    ///             Array data of vector on GPU device.
    ///
    /// \sa         destroy_cusparse_vector,
    ///             create_cusparse_csr_matrix,
    ///             create_cusparse_csc_matrix

    #if defined(USE_CUDA_FP8_E4M3) && (USE_CUDA_FP8_E4M3 == 1)
    template<>
    void create_cusparse_vector<__nv_fp8_e4m3>(
            cusparseDnVecDescr_t& cusparse_vector,
            const LongIndexType vector_size,
            __nv_fp8_e4m3* RESTRICT device_vector)
    {
        // TODO
        throw std::runtime_error("Function not implemented.");
    }
    #endif


    // ======================
    // create cusparse vector (__half)
    // ======================

    /// \brief      A template wrapper for \c cusparseCreateDnVec for the
    ///             \c __half precision data.
    ///
    /// \details    Note that according to the cusparse documentation for the
    ///             function \c cusparseCreateDnVec, it is safe to use
    ///             \c const_cast to cast the input vector.
    ///
    /// \param[out] cusparse_vector
    ///             Reference to CuSparse vector object to be created.
    /// \param[in]  vector_size
    ///             Size of vector array
    /// \param[in]  device_vector
    ///             Array data of vector on GPU device.
    ///
    /// \sa         destroy_cusparse_vector,
    ///             create_cusparse_csr_matrix,
    ///             create_cusparse_csc_matrix

    #if defined(USE_CUDA_FP16) && (USE_CUDA_FP16 == 1)
    template<>
    void create_cusparse_vector<__half>(
            cusparseDnVecDescr_t& cusparse_vector,
            const LongIndexType vector_size,
            __half* RESTRICT device_vector)
    {
        cusparseStatus_t status = cusparseCreateDnVec(
                &cusparse_vector, vector_size, device_vector, CUDA_R_16F);

        assert(status == CUSPARSE_STATUS_SUCCESS);
    }
    #endif


    // ======================
    // create cusparse vector (__nv_bfloat16)
    // ======================

    /// \brief      A template wrapper for \c cusparseCreateDnVec for the
    ///             \c __nv_bfloat16 precision data.
    ///
    /// \details    Note that according to the cusparse documentation for the
    ///             function \c cusparseCreateDnVec, it is safe to use
    ///             \c const_cast to cast the input vector.
    ///
    /// \param[out] cusparse_vector
    ///             Reference to CuSparse vector object to be created.
    /// \param[in]  vector_size
    ///             Size of vector array
    /// \param[in]  device_vector
    ///             Array data of vector on GPU device.
    ///
    /// \sa         destroy_cusparse_vector,
    ///             create_cusparse_csr_matrix,
    ///             create_cusparse_csc_matrix

    #if defined(USE_CUDA_BF16) && (USE_CUDA_BF16 == 1)
    template<>
    void create_cusparse_vector<__nv_bfloat16>(
            cusparseDnVecDescr_t& cusparse_vector,
            const LongIndexType vector_size,
            __nv_bfloat16* RESTRICT device_vector)
    {
        cusparseStatus_t status = cusparseCreateDnVec(
                &cusparse_vector, vector_size, device_vector, CUDA_R_16F);

        assert(status == CUSPARSE_STATUS_SUCCESS);
    }
    #endif


    // ======================
    // create cusparse vector (float)
    // ======================

    /// \brief      A template wrapper for \c cusparseCreateDnVec for the
    ///             \c float precision data.
    ///
    /// \details    Note that according to the cusparse documentation for the
    ///             function \c cusparseCreateDnVec, it is safe to use
    ///             \c const_cast to cast the input vector.
    ///
    /// \param[out] cusparse_vector
    ///             Reference to CuSparse vector object to be created.
    /// \param[in]  vector_size
    ///             Size of vector array
    /// \param[in]  device_vector
    ///             Array data of vector on GPU device.
    ///
    /// \sa         destroy_cusparse_vector,
    ///             create_cusparse_csr_matrix,
    ///             create_cusparse_csc_matrix

    #if defined(USE_CUDA_FP32) && (USE_CUDA_FP32 == 1)
    template<>
    void create_cusparse_vector<float>(
            cusparseDnVecDescr_t& cusparse_vector,
            const LongIndexType vector_size,
            float* RESTRICT device_vector)
    {
        cusparseStatus_t status = cusparseCreateDnVec(
                &cusparse_vector, vector_size, device_vector, CUDA_R_32F);

        assert(status == CUSPARSE_STATUS_SUCCESS);
    }
    #endif


    // ======================
    // create cusparse vector (double)
    // ======================

    /// \brief      A template wrapper for \c cusparseCreateDnVec for the
    ///             \c double precision data.
    ///
    /// \details    Note that according to the cusparse documentation for the
    ///             function \c cusparseCreateDnVec, it is safe to use
    ///             \c const_cast to cast the input vector.
    ///
    /// \param[out] cusparse_vector
    ///             Reference to CuSparse vector object to be created.
    /// \param[in]  vector_size
    ///             Size of vector array
    /// \param[in]  device_vector
    ///             Array data of vector on GPU device.
    ///
    /// \sa         destroy_cusparse_vector,
    ///             create_cusparse_csr_matrix,
    ///             create_cusparse_csc_matrix

    #if defined(USE_CUDA_FP64) && (USE_CUDA_FP64 == 1)
    template<>
    void create_cusparse_vector<double>(
            cusparseDnVecDescr_t& cusparse_vector,
            const LongIndexType vector_size,
            double* RESTRICT device_vector)
    {
        cusparseStatus_t status = cusparseCreateDnVec(
                &cusparse_vector, vector_size, device_vector, CUDA_R_64F);

        assert(status == CUSPARSE_STATUS_SUCCESS);
    }
    #endif


    // =======================
    // destroy cusparse matrix
    // =======================

    /// \brief      Destroy cusparse matrix.
    ///
    /// \details    This function is a wrapper for \c cusparseDestroySpMat.
    ///
    /// \param[out] cusparse_matrix
    ///             Reference to CuSparse matrix object to be destroyed.
    ///
    /// \sa         destroy_cusparse_vector,
    ///             create_cusparse_csr_matrix,
    ///             create_cusparse_csc_matrix

    void destroy_cusparse_matrix(
            cusparseSpMatDescr_t& cusparse_matrix)
    {
        cusparseStatus_t status = cusparseDestroySpMat(cusparse_matrix);
        assert(status == CUSPARSE_STATUS_SUCCESS);
    }


    // =======================
    // destroy cusparse vector
    // =======================

    /// \brief      Destroys cusparse vector.
    ///
    /// \details    This function is a wrapper for \c cusparseDestroyDnVec.
    ///
    /// \param[out] cusparse_vector
    ///             Reference to CuSparse vector object to be destroyed.
    ///
    /// \sa         destroy_cusparse_matrix,
    ///             create_cusparse_vector

    void destroy_cusparse_vector(
            cusparseDnVecDescr_t& cusparse_vector)
    {
        cusparseStatus_t status = cusparseDestroyDnVec(cusparse_vector);
        assert(status == CUSPARSE_STATUS_SUCCESS);
    }


    // ===========================
    // cusparse matrix buffer size (__nv_fp8_e5m2)
    // ===========================

    /// \brief      A template wrapper for \c cusparseSpMV_bufferSize for
    ///             \c __nv_fp8_e5m2 precision data. This function determines
    ///             the buffer size needed for matrix-vector multiplication
    ///             using \c cusparseSpMV. The output is \c buffer_size
    ///             variable.
    ///        
    /// \details    Note that this function uses mixed-precision computation
    ///             where the matrix and vectors are __nv_fp8_e5m2 type while
    ///             the compute type is CUDA_R_32.
    ///
    /// \param[in]  cusparse_handle
    ///             Handle to the CuSparse library context
    /// \param[in]  cusparse_operation
    ///             Type of matrix operation. For non-transpose operation, this
    ///             should be set to \c CUSPARSE_OPERATION_NON_TRANSPOSE  and
    ///             for transpose operation, this should be set to
    ///             \c CUSPARSE_OPERATION_TRANSPOSE.
    /// \param[in]  alpha
    ///             The scalar parameter \f$ \alpha \f$ in matrix-vector
    ///             product.
    /// \param[in]  cusparse_matrix
    ///             Cusparse object for matrix \f$ \mathbf{A} \f$.
    /// \param[in]  cusparse_input_vector
    ///             Cusparse object for vector \f$ \boldsymbol{x} \f$.
    /// \param[in]  beta
    ///             The scalar parameter \f$ \beta \f$ in matrix-vector
    ///             product.
    /// \param[in]  cusparse_output_vector
    ///             Cusparse object for vector \f$ \boldsymbol{y} \f$.
    /// \param[in]  algorithm
    ///             Algorithm for the computation. Possible values can be
    ///             \c CUSPARSE_SPMV_ALG_DEFAULT, \c CUSPARSE_SPMV_CSR_ALG1,
    ///             and \c CUSPARSE_SPMV_CSR_ALG2.
    /// \param[out] buffer_size
    ///             The size of buffer needed for computation.
    ///
    /// \sa         cusparse_matvec

    #if defined(USE_CUDA_FP8_E5M2) && (USE_CUDA_FP8_E5M2 == 1)
    template<>
    void cusparse_matrix_buffer_size<__nv_fp8_e5m2>(
            cusparseHandle_t cusparse_handle,
            cusparseOperation_t cusparse_operation,
            const __nv_fp8_e5m2 alpha,
            cusparseSpMatDescr_t cusparse_matrix,
            cusparseDnVecDescr_t cusparse_input_vector,
            const __nv_fp8_e5m2 beta,
            cusparseDnVecDescr_t cusparse_output_vector,
            cusparseSpMVAlg_t algorithm,
            size_t* buffer_size)
    {
        // TODO
        throw std::runtime_error("Function not implemented.");
    }
    #endif


    // ===========================
    // cusparse matrix buffer size (__nv_fp8_e4m3)
    // ===========================

    /// \brief      A template wrapper for \c cusparseSpMV_bufferSize for
    ///             \c __nv_fp8_e4m3 precision data. This function determines
    ///             the buffer size needed for matrix-vector multiplication
    ///             using \c cusparseSpMV. The output is \c buffer_size
    ///             variable.
    ///        
    /// \details    Note that this function uses mixed-precision computation
    ///             where the matrix and vectors are __nv_fp8_e4m3 type while
    ///             the compute type is CUDA_R_32.
    ///
    /// \param[in]  cusparse_handle
    ///             Handle to the CuSparse library context
    /// \param[in]  cusparse_operation
    ///             Type of matrix operation. For non-transpose operation, this
    ///             should be set to \c CUSPARSE_OPERATION_NON_TRANSPOSE  and
    ///             for transpose operation, this should be set to
    ///             \c CUSPARSE_OPERATION_TRANSPOSE.
    /// \param[in]  alpha
    ///             The scalar parameter \f$ \alpha \f$ in matrix-vector
    ///             product.
    /// \param[in]  cusparse_matrix
    ///             Cusparse object for matrix \f$ \mathbf{A} \f$.
    /// \param[in]  cusparse_input_vector
    ///             Cusparse object for vector \f$ \boldsymbol{x} \f$.
    /// \param[in]  beta
    ///             The scalar parameter \f$ \beta \f$ in matrix-vector
    ///             product.
    /// \param[in]  cusparse_output_vector
    ///             Cusparse object for vector \f$ \boldsymbol{y} \f$.
    /// \param[in]  algorithm
    ///             Algorithm for the computation. Possible values can be
    ///             \c CUSPARSE_SPMV_ALG_DEFAULT, \c CUSPARSE_SPMV_CSR_ALG1,
    ///             and \c CUSPARSE_SPMV_CSR_ALG2.
    /// \param[out] buffer_size
    ///             The size of buffer needed for computation.
    ///
    /// \sa         cusparse_matvec

    #if defined(USE_CUDA_FP8_E4M3) && (USE_CUDA_FP8_E4M3 == 1)
    template<>
    void cusparse_matrix_buffer_size<__nv_fp8_e4m3>(
            cusparseHandle_t cusparse_handle,
            cusparseOperation_t cusparse_operation,
            const __nv_fp8_e4m3 alpha,
            cusparseSpMatDescr_t cusparse_matrix,
            cusparseDnVecDescr_t cusparse_input_vector,
            const __nv_fp8_e4m3 beta,
            cusparseDnVecDescr_t cusparse_output_vector,
            cusparseSpMVAlg_t algorithm,
            size_t* buffer_size)
    {
        // TODO
        throw std::runtime_error("Function not implemented.");
    }
    #endif


    // ===========================
    // cusparse matrix buffer size (__half)
    // ===========================

    /// \brief      A template wrapper for \c cusparseSpMV_bufferSize for
    ///             \c __half precision data. This function determines the
    ///             buffer size needed for matrix-vector multiplication using
    ///             \c cusparseSpMV. The output is \c buffer_size variable.
    ///        
    /// \details    Note that this function uses mixed-precision computation
    ///             where the matrix and vectors are __half type while the
    ///             compute type is CUDA_R_32.
    ///
    /// \param[in]  cusparse_handle
    ///             Handle to the CuSparse library context
    /// \param[in]  cusparse_operation
    ///             Type of matrix operation. For non-transpose operation, this
    ///             should be set to \c CUSPARSE_OPERATION_NON_TRANSPOSE  and
    ///             for transpose operation, this should be set to
    ///             \c CUSPARSE_OPERATION_TRANSPOSE.
    /// \param[in]  alpha
    ///             The scalar parameter \f$ \alpha \f$ in matrix-vector
    ///             product.
    /// \param[in]  cusparse_matrix
    ///             Cusparse object for matrix \f$ \mathbf{A} \f$.
    /// \param[in]  cusparse_input_vector
    ///             Cusparse object for vector \f$ \boldsymbol{x} \f$.
    /// \param[in]  beta
    ///             The scalar parameter \f$ \beta \f$ in matrix-vector
    ///             product.
    /// \param[in]  cusparse_output_vector
    ///             Cusparse object for vector \f$ \boldsymbol{y} \f$.
    /// \param[in]  algorithm
    ///             Algorithm for the computation. Possible values can be
    ///             \c CUSPARSE_SPMV_ALG_DEFAULT, \c CUSPARSE_SPMV_CSR_ALG1,
    ///             and \c CUSPARSE_SPMV_CSR_ALG2.
    /// \param[out] buffer_size
    ///             The size of buffer needed for computation.
    ///
    /// \sa         cusparse_matvec

    #if defined(USE_CUDA_FP16) && (USE_CUDA_FP16 == 1)
    template<>
    void cusparse_matrix_buffer_size<__half>(
            cusparseHandle_t cusparse_handle,
            cusparseOperation_t cusparse_operation,
            const __half alpha,
            cusparseSpMatDescr_t cusparse_matrix,
            cusparseDnVecDescr_t cusparse_input_vector,
            const __half beta,
            cusparseDnVecDescr_t cusparse_output_vector,
            cusparseSpMVAlg_t algorithm,
            size_t* buffer_size)
    {
        cusparseStatus_t status = cusparseSpMV_bufferSize(
                cusparse_handle, cusparse_operation, &alpha, cusparse_matrix,
                cusparse_input_vector, &beta, cusparse_output_vector,
                CUDA_R_32F, algorithm, buffer_size);

        assert(status == CUSPARSE_STATUS_SUCCESS);
    }
    #endif

    // ===========================
    // cusparse matrix buffer size (__nv_bfloat16)
    // ===========================

    /// \brief      A template wrapper for \c cusparseSpMV_bufferSize for
    ///             \c __nv_bfloat16 precision data. This function determines
    ///             the buffer size needed for matrix-vector multiplication
    ///             using \c cusparseSpMV. The output is \c buffer_size
    ///             variable.
    ///        
    /// \details    Note that this function uses mixed-precision computation
    ///             where the matrix and vectors are __nv_bfloat16 type while
    ///             the compute type is CUDA_R_32.
    ///
    /// \param[in]  cusparse_handle
    ///             Handle to the CuSparse library context
    /// \param[in]  cusparse_operation
    ///             Type of matrix operation. For non-transpose operation, this
    ///             should be set to \c CUSPARSE_OPERATION_NON_TRANSPOSE  and
    ///             for transpose operation, this should be set to
    ///             \c CUSPARSE_OPERATION_TRANSPOSE.
    /// \param[in]  alpha
    ///             The scalar parameter \f$ \alpha \f$ in matrix-vector
    ///             product.
    /// \param[in]  cusparse_matrix
    ///             Cusparse object for matrix \f$ \mathbf{A} \f$.
    /// \param[in]  cusparse_input_vector
    ///             Cusparse object for vector \f$ \boldsymbol{x} \f$.
    /// \param[in]  beta
    ///             The scalar parameter \f$ \beta \f$ in matrix-vector
    ///             product.
    /// \param[in]  cusparse_output_vector
    ///             Cusparse object for vector \f$ \boldsymbol{y} \f$.
    /// \param[in]  algorithm
    ///             Algorithm for the computation. Possible values can be
    ///             \c CUSPARSE_SPMV_ALG_DEFAULT, \c CUSPARSE_SPMV_CSR_ALG1,
    ///             and \c CUSPARSE_SPMV_CSR_ALG2.
    /// \param[out] buffer_size
    ///             The size of buffer needed for computation.
    ///
    /// \sa         cusparse_matvec

    #if defined(USE_CUDA_BF16) && (USE_CUDA_BF16 == 1)
    template<>
    void cusparse_matrix_buffer_size<__nv_bfloat16>(
            cusparseHandle_t cusparse_handle,
            cusparseOperation_t cusparse_operation,
            const __nv_bfloat16 alpha,
            cusparseSpMatDescr_t cusparse_matrix,
            cusparseDnVecDescr_t cusparse_input_vector,
            const __nv_bfloat16 beta,
            cusparseDnVecDescr_t cusparse_output_vector,
            cusparseSpMVAlg_t algorithm,
            size_t* buffer_size)
    {
        cusparseStatus_t status = cusparseSpMV_bufferSize(
                cusparse_handle, cusparse_operation, &alpha, cusparse_matrix,
                cusparse_input_vector, &beta, cusparse_output_vector,
                CUDA_R_32F, algorithm, buffer_size);

        assert(status == CUSPARSE_STATUS_SUCCESS);
    }
    #endif


    // ===========================
    // cusparse matrix buffer size (float)
    // ===========================

    /// \brief      A template wrapper for \c cusparseSpMV_bufferSize for
    ///             \c float precision data. This function determines the
    ///             buffer size needed for matrix-vector multiplication using
    ///             \c cusparseSpMV. The output is \c buffer_size variable.
    ///
    /// \param[in]  cusparse_handle
    ///             Handle to the CuSparse library context
    /// \param[in]  cusparse_operation
    ///             Type of matrix operation. For non-transpose operation, this
    ///             should be set to \c CUSPARSE_OPERATION_NON_TRANSPOSE  and
    ///             for transpose operation, this should be set to
    ///             \c CUSPARSE_OPERATION_TRANSPOSE.
    /// \param[in]  alpha
    ///             The scalar parameter \f$ \alpha \f$ in matrix-vector
    ///             product.
    /// \param[in]  cusparse_matrix
    ///             Cusparse object for matrix \f$ \mathbf{A} \f$.
    /// \param[in]  cusparse_input_vector
    ///             Cusparse object for vector \f$ \boldsymbol{x} \f$.
    /// \param[in]  beta
    ///             The scalar parameter \f$ \beta \f$ in matrix-vector
    ///             product.
    /// \param[in]  cusparse_output_vector
    ///             Cusparse object for vector \f$ \boldsymbol{y} \f$.
    /// \param[in]  algorithm
    ///             Algorithm for the computation. Possible values can be
    ///             \c CUSPARSE_SPMV_ALG_DEFAULT, \c CUSPARSE_SPMV_CSR_ALG1,
    ///             and \c CUSPARSE_SPMV_CSR_ALG2.
    /// \param[out] buffer_size
    ///             The size of buffer needed for computation.
    ///
    /// \sa         cusparse_matvec

    #if defined(USE_CUDA_FP32) && (USE_CUDA_FP32 == 1)
    template<>
    void cusparse_matrix_buffer_size<float>(
            cusparseHandle_t cusparse_handle,
            cusparseOperation_t cusparse_operation,
            const float alpha,
            cusparseSpMatDescr_t cusparse_matrix,
            cusparseDnVecDescr_t cusparse_input_vector,
            const float beta,
            cusparseDnVecDescr_t cusparse_output_vector,
            cusparseSpMVAlg_t algorithm,
            size_t* buffer_size)
    {
        cusparseStatus_t status = cusparseSpMV_bufferSize(
                cusparse_handle, cusparse_operation, &alpha, cusparse_matrix,
                cusparse_input_vector, &beta, cusparse_output_vector,
                CUDA_R_32F, algorithm, buffer_size);

        assert(status == CUSPARSE_STATUS_SUCCESS);
    }
    #endif


    // ===========================
    // cusparse matrix buffer size (double)
    // ===========================

    /// \brief      A template wrapper for \c cusparseSpMV_bufferSize for
    ///             \c double precision data. This function determines the
    ///             buffer size needed for matrix-vector multiplication using
    ///             \c cusparseSpMV. The output is \c buffer_size variable.
    ///        
    /// \param[in]  cusparse_handle
    ///             Handle to the CuSparse library context
    /// \param[in]  cusparse_operation
    ///             Type of matrix operation. For non-transpose operation, this
    ///             should be set to \c CUSPARSE_OPERATION_NON_TRANSPOSE  and
    ///             for transpose operation, this should be set to
    ///             \c CUSPARSE_OPERATION_TRANSPOSE.
    /// \param[in]  alpha
    ///             The scalar parameter \f$ \alpha \f$ in matrix-vector
    ///             product.
    /// \param[in]  cusparse_matrix
    ///             Cusparse object for matrix \f$ \mathbf{A} \f$.
    /// \param[in]  cusparse_input_vector
    ///             Cusparse object for vector \f$ \boldsymbol{x} \f$.
    /// \param[in]  beta
    ///             The scalar parameter \f$ \beta \f$ in matrix-vector
    ///             product.
    /// \param[in]  cusparse_output_vector
    ///             Cusparse object for vector \f$ \boldsymbol{y} \f$.
    /// \param[in]  algorithm
    ///             Algorithm for the computation. Possible values can be
    ///             \c CUSPARSE_SPMV_ALG_DEFAULT, \c CUSPARSE_SPMV_CSR_ALG1,
    ///             and \c CUSPARSE_SPMV_CSR_ALG2.
    /// \param[out] buffer_size
    ///             The size of buffer needed for computation.
    ///
    /// \sa         cusparse_matvec

    #if defined(USE_CUDA_FP64) && (USE_CUDA_FP64 == 1)
    template<>
    void cusparse_matrix_buffer_size<double>(
            cusparseHandle_t cusparse_handle,
            cusparseOperation_t cusparse_operation,
            const double alpha,
            cusparseSpMatDescr_t cusparse_matrix,
            cusparseDnVecDescr_t cusparse_input_vector,
            const double beta,
            cusparseDnVecDescr_t cusparse_output_vector,
            cusparseSpMVAlg_t algorithm,
            size_t* buffer_size)
    {
        cusparseStatus_t status = cusparseSpMV_bufferSize(
                cusparse_handle, cusparse_operation, &alpha, cusparse_matrix,
                cusparse_input_vector, &beta, cusparse_output_vector,
                CUDA_R_64F, algorithm, buffer_size);

        assert(status == CUSPARSE_STATUS_SUCCESS);
    }
    #endif


    // ===============
    // cusparse matvec (__nv_fp8_e5m2)
    // ===============

    /// \brief      A wrapper for \c cusparseSpMV to perform sparse
    ///             matrix-vector multiplication using \c __nv_fp8_e5m2
    ///             precision data.
    ///
    /// \details    Note that this function uses mixed-precision computation
    ///             where the matrix and vectors are __nv_fp8_e5m2 type while
    ///             the compute type is CUDA_R_32.
    ///
    /// \param[in]  cusparse_handle
    ///             Handle to the CuSparse library context
    /// \param[in]  cusparse_operation
    ///             Type of matrix operation. For non-transpose operation, this
    ///             should be set to \c CUSPARSE_OPERATION_NON_TRANSPOSE  and
    ///             for transpose operation, this should be set to
    ///             \c CUSPARSE_OPERATION_TRANSPOSE.
    /// \param[in]  alpha
    ///             The scalar parameter \f$ \alpha \f$ in matrix-vector
    ///             product.
    /// \param[in]  cusparse_matrix
    ///             Cusparse object for matrix \f$ \mathbf{A} \f$.
    /// \param[in]  cusparse_input_vector
    ///             Cusparse object for vector \f$ \boldsymbol{x} \f$.
    /// \param[in]  beta
    ///             The scalar parameter \f$ \beta \f$ in matrix-vector
    ///             product.
    /// \param[out] cusparse_output_vector
    ///             Cusparse object for vector \f$ \boldsymbol{y} \f$.
    /// \param[in]  algorithm
    ///             Algorithm for the computation. Possible values can be
    ///             \c CUSPARSE_SPMV_ALG_DEFAULT, \c CUSPARSE_SPMV_CSR_ALG1,
    ///             and \c CUSPARSE_SPMV_CSR_ALG2.
    /// \param[in]  external_buffer
    ///             Buffer on GPU device needed for computation.
    ///
    /// \sa         cusparse_matrix_buffer_size

    #if defined(USE_CUDA_FP8_E5M2) && (USE_CUDA_FP8_E5M2 == 1)
    template<>
    void cusparse_matvec<__nv_fp8_e5m2>(
            cusparseHandle_t cusparse_handle,
            cusparseOperation_t cusparse_operation,
            const __nv_fp8_e5m2 alpha,
            cusparseSpMatDescr_t cusparse_matrix,
            cusparseDnVecDescr_t cusparse_input_vector,
            const __nv_fp8_e5m2 beta,
            cusparseDnVecDescr_t cusparse_output_vector,
            cusparseSpMVAlg_t algorithm,
            void* external_buffer)
    {
        // TODO
        throw std::runtime_error("Function not implemented.");
    }
    #endif


    // ===============
    // cusparse matvec (__nv_fp8_e4m3)
    // ===============

    /// \brief      A wrapper for \c cusparseSpMV to perform sparse
    ///             matrix-vector multiplication using \c __nv_fp8_e4m3
    ///             precision data.
    ///
    /// \details    Note that this function uses mixed-precision computation
    ///             where the matrix and vectors are __nv_fp8_e4m3 type while
    ///             the compute type is CUDA_R_32.
    ///
    /// \param[in]  cusparse_handle
    ///             Handle to the CuSparse library context
    /// \param[in]  cusparse_operation
    ///             Type of matrix operation. For non-transpose operation, this
    ///             should be set to \c CUSPARSE_OPERATION_NON_TRANSPOSE  and
    ///             for transpose operation, this should be set to
    ///             \c CUSPARSE_OPERATION_TRANSPOSE.
    /// \param[in]  alpha
    ///             The scalar parameter \f$ \alpha \f$ in matrix-vector
    ///             product.
    /// \param[in]  cusparse_matrix
    ///             Cusparse object for matrix \f$ \mathbf{A} \f$.
    /// \param[in]  cusparse_input_vector
    ///             Cusparse object for vector \f$ \boldsymbol{x} \f$.
    /// \param[in]  beta
    ///             The scalar parameter \f$ \beta \f$ in matrix-vector
    ///             product.
    /// \param[out] cusparse_output_vector
    ///             Cusparse object for vector \f$ \boldsymbol{y} \f$.
    /// \param[in]  algorithm
    ///             Algorithm for the computation. Possible values can be
    ///             \c CUSPARSE_SPMV_ALG_DEFAULT, \c CUSPARSE_SPMV_CSR_ALG1,
    ///             and \c CUSPARSE_SPMV_CSR_ALG2.
    /// \param[in]  external_buffer
    ///             Buffer on GPU device needed for computation.
    ///
    /// \sa         cusparse_matrix_buffer_size

    #if defined(USE_CUDA_FP8_E4M3) && (USE_CUDA_FP8_E4M3 == 1)
    template<>
    void cusparse_matvec<__nv_fp8_e4m3>(
            cusparseHandle_t cusparse_handle,
            cusparseOperation_t cusparse_operation,
            const __nv_fp8_e4m3 alpha,
            cusparseSpMatDescr_t cusparse_matrix,
            cusparseDnVecDescr_t cusparse_input_vector,
            const __nv_fp8_e4m3 beta,
            cusparseDnVecDescr_t cusparse_output_vector,
            cusparseSpMVAlg_t algorithm,
            void* external_buffer)
    {
        // TODO
        throw std::runtime_error("Function not implemented.");
    }
    #endif


    // ===============
    // cusparse matvec (__half)
    // ===============

    /// \brief      A wrapper for \c cusparseSpMV to perform sparse
    ///             matrix-vector multiplication using \c __half precision
    ///             data.
    ///
    /// \details    Note that this function uses mixed-precision computation
    ///             where the matrix and vectors are __half type while the
    ///             compute type is CUDA_R_32.
    ///
    /// \param[in]  cusparse_handle
    ///             Handle to the CuSparse library context
    /// \param[in]  cusparse_operation
    ///             Type of matrix operation. For non-transpose operation, this
    ///             should be set to \c CUSPARSE_OPERATION_NON_TRANSPOSE  and
    ///             for transpose operation, this should be set to
    ///             \c CUSPARSE_OPERATION_TRANSPOSE.
    /// \param[in]  alpha
    ///             The scalar parameter \f$ \alpha \f$ in matrix-vector
    ///             product.
    /// \param[in]  cusparse_matrix
    ///             Cusparse object for matrix \f$ \mathbf{A} \f$.
    /// \param[in]  cusparse_input_vector
    ///             Cusparse object for vector \f$ \boldsymbol{x} \f$.
    /// \param[in]  beta
    ///             The scalar parameter \f$ \beta \f$ in matrix-vector
    ///             product.
    /// \param[out] cusparse_output_vector
    ///             Cusparse object for vector \f$ \boldsymbol{y} \f$.
    /// \param[in]  algorithm
    ///             Algorithm for the computation. Possible values can be
    ///             \c CUSPARSE_SPMV_ALG_DEFAULT, \c CUSPARSE_SPMV_CSR_ALG1,
    ///             and \c CUSPARSE_SPMV_CSR_ALG2.
    /// \param[in]  external_buffer
    ///             Buffer on GPU device needed for computation.
    ///
    /// \sa         cusparse_matrix_buffer_size

    #if defined(USE_CUDA_FP16) && (USE_CUDA_FP16 == 1)
    template<>
    void cusparse_matvec<__half>(
            cusparseHandle_t cusparse_handle,
            cusparseOperation_t cusparse_operation,
            const __half alpha,
            cusparseSpMatDescr_t cusparse_matrix,
            cusparseDnVecDescr_t cusparse_input_vector,
            const __half beta,
            cusparseDnVecDescr_t cusparse_output_vector,
            cusparseSpMVAlg_t algorithm,
            void* external_buffer)
    {
        cusparseStatus_t status = cusparseSpMV(cusparse_handle,
                                               cusparse_operation, &alpha,
                                               cusparse_matrix,
                                               cusparse_input_vector, &beta,
                                               cusparse_output_vector,
                                               CUDA_R_32F, algorithm,
                                               external_buffer);

        assert(status == CUSPARSE_STATUS_SUCCESS);
    }
    #endif


    // ===============
    // cusparse matvec (__nv_bfloat16)
    // ===============

    /// \brief      A wrapper for \c cusparseSpMV to perform sparse
    ///             matrix-vector multiplication using \c __nv_bfloat16
    ///             precision data.
    ///
    /// \details    Note that this function uses mixed-precision computation
    ///             where the matrix and vectors are __nv_bfloat16 type while
    ///             the compute type is CUDA_R_32.
    ///
    /// \param[in]  cusparse_handle
    ///             Handle to the CuSparse library context
    /// \param[in]  cusparse_operation
    ///             Type of matrix operation. For non-transpose operation, this
    ///             should be set to \c CUSPARSE_OPERATION_NON_TRANSPOSE  and
    ///             for transpose operation, this should be set to
    ///             \c CUSPARSE_OPERATION_TRANSPOSE.
    /// \param[in]  alpha
    ///             The scalar parameter \f$ \alpha \f$ in matrix-vector
    ///             product.
    /// \param[in]  cusparse_matrix
    ///             Cusparse object for matrix \f$ \mathbf{A} \f$.
    /// \param[in]  cusparse_input_vector
    ///             Cusparse object for vector \f$ \boldsymbol{x} \f$.
    /// \param[in]  beta
    ///             The scalar parameter \f$ \beta \f$ in matrix-vector
    ///             product.
    /// \param[out] cusparse_output_vector
    ///             Cusparse object for vector \f$ \boldsymbol{y} \f$.
    /// \param[in]  algorithm
    ///             Algorithm for the computation. Possible values can be
    ///             \c CUSPARSE_SPMV_ALG_DEFAULT, \c CUSPARSE_SPMV_CSR_ALG1,
    ///             and \c CUSPARSE_SPMV_CSR_ALG2.
    /// \param[in]  external_buffer
    ///             Buffer on GPU device needed for computation.
    ///
    /// \sa         cusparse_matrix_buffer_size

    #if defined(USE_CUDA_BF16) && (USE_CUDA_BF16 == 1)
    template<>
    void cusparse_matvec<__nv_bfloat16>(
            cusparseHandle_t cusparse_handle,
            cusparseOperation_t cusparse_operation,
            const __nv_bfloat16 alpha,
            cusparseSpMatDescr_t cusparse_matrix,
            cusparseDnVecDescr_t cusparse_input_vector,
            const __nv_bfloat16 beta,
            cusparseDnVecDescr_t cusparse_output_vector,
            cusparseSpMVAlg_t algorithm,
            void* external_buffer)
    {
        cusparseStatus_t status = cusparseSpMV(cusparse_handle,
                                               cusparse_operation, &alpha,
                                               cusparse_matrix,
                                               cusparse_input_vector, &beta,
                                               cusparse_output_vector,
                                               CUDA_R_32F, algorithm,
                                               external_buffer);

        assert(status == CUSPARSE_STATUS_SUCCESS);
    }
    #endif


    // ===============
    // cusparse matvec (float)
    // ===============

    /// \brief      A wrapper for \c cusparseSpMV to perform sparse
    ///             matrix-vector multiplication using \c float precision
    ///             data.
    ///
    /// \param[in]  cusparse_handle
    ///             Handle to the CuSparse library context
    /// \param[in]  cusparse_operation
    ///             Type of matrix operation. For non-transpose operation, this
    ///             should be set to \c CUSPARSE_OPERATION_NON_TRANSPOSE  and
    ///             for transpose operation, this should be set to
    ///             \c CUSPARSE_OPERATION_TRANSPOSE.
    /// \param[in]  alpha
    ///             The scalar parameter \f$ \alpha \f$ in matrix-vector
    ///             product.
    /// \param[in]  cusparse_matrix
    ///             Cusparse object for matrix \f$ \mathbf{A} \f$.
    /// \param[in]  cusparse_input_vector
    ///             Cusparse object for vector \f$ \boldsymbol{x} \f$.
    /// \param[in]  beta
    ///             The scalar parameter \f$ \beta \f$ in matrix-vector
    ///             product.
    /// \param[out] cusparse_output_vector
    ///             Cusparse object for vector \f$ \boldsymbol{y} \f$.
    /// \param[in]  algorithm
    ///             Algorithm for the computation. Possible values can be
    ///             \c CUSPARSE_SPMV_ALG_DEFAULT, \c CUSPARSE_SPMV_CSR_ALG1,
    ///             and \c CUSPARSE_SPMV_CSR_ALG2.
    /// \param[in]  external_buffer
    ///             Buffer on GPU device needed for computation.
    ///
    /// \sa         cusparse_matrix_buffer_size

    #if defined(USE_CUDA_FP32) && (USE_CUDA_FP32 == 1)
    template<>
    void cusparse_matvec<float>(
            cusparseHandle_t cusparse_handle,
            cusparseOperation_t cusparse_operation,
            const float alpha,
            cusparseSpMatDescr_t cusparse_matrix,
            cusparseDnVecDescr_t cusparse_input_vector,
            const float beta,
            cusparseDnVecDescr_t cusparse_output_vector,
            cusparseSpMVAlg_t algorithm,
            void* external_buffer)
    {
        cusparseStatus_t status = cusparseSpMV(cusparse_handle,
                                               cusparse_operation, &alpha,
                                               cusparse_matrix,
                                               cusparse_input_vector, &beta,
                                               cusparse_output_vector,
                                               CUDA_R_32F, algorithm,
                                               external_buffer);

        assert(status == CUSPARSE_STATUS_SUCCESS);
    }
    #endif


    // ===============
    // cusparse matvec (double)
    // ===============

    /// \brief      A wrapper for \c cusparseSpMV to perform sparse
    ///             matrix-vector multiplication using \c double precision
    ///             data.
    ///
    /// \param[in]  cusparse_handle
    ///             Handle to the CuSparse library context
    /// \param[in]  cusparse_operation
    ///             Type of matrix operation. For non-transpose operation, this
    ///             should be set to \c CUSPARSE_OPERATION_NON_TRANSPOSE  and
    ///             for transpose operation, this should be set to
    ///             \c CUSPARSE_OPERATION_TRANSPOSE.
    /// \param[in]  alpha
    ///             The scalar parameter \f$ \alpha \f$ in matrix-vector
    ///             product.
    /// \param[in]  cusparse_matrix
    ///             Cusparse object for matrix \f$ \mathbf{A} \f$.
    /// \param[in]  cusparse_input_vector
    ///             Cusparse object for vector \f$ \boldsymbol{x} \f$.
    /// \param[in]  beta
    ///             The scalar parameter \f$ \beta \f$ in matrix-vector
    ///             product.
    /// \param[out] cusparse_output_vector
    ///             Cusparse object for vector \f$ \boldsymbol{y} \f$.
    /// \param[in]  algorithm
    ///             Algorithm for the computation. Possible values can be
    ///             \c CUSPARSE_SPMV_ALG_DEFAULT, \c CUSPARSE_SPMV_CSR_ALG1,
    ///             and \c CUSPARSE_SPMV_CSR_ALG2.
    /// \param[in]  external_buffer
    ///             Buffer on GPU device needed for computation.
    ///
    /// \sa         cusparse_matrix_buffer_size

    #if defined(USE_CUDA_FP64) && (USE_CUDA_FP64 == 1)
    template<>
    void cusparse_matvec<double>(
            cusparseHandle_t cusparse_handle,
            cusparseOperation_t cusparse_operation,
            const double alpha,
            cusparseSpMatDescr_t cusparse_matrix,
            cusparseDnVecDescr_t cusparse_input_vector,
            const double beta,
            cusparseDnVecDescr_t cusparse_output_vector,
            cusparseSpMVAlg_t algorithm,
            void* external_buffer)
    {
        cusparseStatus_t status = cusparseSpMV(cusparse_handle,
                                               cusparse_operation, &alpha,
                                               cusparse_matrix,
                                               cusparse_input_vector, &beta,
                                               cusparse_output_vector,
                                               CUDA_R_64F, algorithm,
                                               external_buffer);

        assert(status == CUSPARSE_STATUS_SUCCESS);
    }
    #endif

}  // namespace cusparse_api
