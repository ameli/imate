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

#include "./cusparse_interface.h"
#include <cassert>  // assert

// ==================
// cusparse interface
// ==================

/// \note      The implementation in the \c cu file is wrapped inside the
///            namepsace clause. This is not necessary in general, however, it
///            is needed to avoid the old gcc compiler error (this is a gcc
///            bug) which complains "no instance of function template matches
///            the argument list const float".

namespace cusparse_interface
{

    // ======================
    // create cusparse matrix (float)
    // ======================

    /// \brief A template wrapper for \c cusparseSpMatDescr_t for the \c float
    ///        precision data.

    template<>
    void create_cusparse_matrix<float>(
            cusparseSpMatDescr_t& cusparse_matrix,
            const LongIndexType num_rows,
            const LongIndexType num_columns,
            const LongIndexType nnz,
            float* device_A_data,
            LongIndexType* device_A_indices,
            LongIndexType* device_A_index_pointer)
    {
        cusparseStatus_t status = cusparseCreateCsr(
                &cusparse_matrix, num_rows, num_columns, nnz,
                device_A_index_pointer, device_A_indices, device_A_data,
                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F);

        assert(status == CUSPARSE_STATUS_SUCCESS);
    }


    // ======================
    // create cusparse matrix (double)
    // ======================

    /// \brief A template wrapper for \c cusparseSpMatDescr_t for the \c double
    ///        precision data.

    template<>
    void create_cusparse_matrix<double>(
            cusparseSpMatDescr_t& cusparse_matrix,
            const LongIndexType num_rows,
            const LongIndexType num_columns,
            const LongIndexType nnz,
            double* device_A_data,
            LongIndexType* device_A_indices,
            LongIndexType* device_A_index_pointer)
    {
        cusparseStatus_t status = cusparseCreateCsr(
                &cusparse_matrix, num_rows, num_columns, nnz,
                device_A_index_pointer, device_A_indices, device_A_data,
                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                CUSPARSE_INDEX_BASE_ZERO, CUDA_R_64F);

        assert(status == CUSPARSE_STATUS_SUCCESS);
    }


    // ======================
    // create cusparse vector (float)
    // ======================

    /// \brief   A template wrapper for \c cusparseDnVecDescr_t for the
    ///          \c float precision data.
    ///
    /// \details Note that according to the cusparse documentation for the
    ///          function \c cusparseCreateDnVec, it is safe to use
    ///          \c const_cast to cast the input vector.

    template<>
    void create_cusparse_vector<float>(
            cusparseDnVecDescr_t& cusparse_vector,
            const LongIndexType vector_size,
            float* device_vector)
    {
        cusparseStatus_t status = cusparseCreateDnVec(
                &cusparse_vector, vector_size, device_vector, CUDA_R_32F);

        assert(status == CUSPARSE_STATUS_SUCCESS);
    }


    // ======================
    // create cusparse vector (double)
    // ======================

    /// \brief   A template wrapper for \c cusparseDnVecDescr_t for the
    ///          \c double precision data.
    ///
    /// \details Note that according to the cusparse documentation for the
    ///          function \c cusparseCreateDnVec, it is safe to use
    ///          \c const_cast to cast the input vector.

    template<>
    void create_cusparse_vector<double>(
            cusparseDnVecDescr_t& cusparse_vector,
            const LongIndexType vector_size,
            double* device_vector)
    {
        cusparseStatus_t status = cusparseCreateDnVec(
                &cusparse_vector, vector_size, device_vector, CUDA_R_64F);

        assert(status == CUSPARSE_STATUS_SUCCESS);
    }


    // =======================
    // destroy cusparse matrix
    // =======================

    /// \brief Destroys cusparse matrix.
    ///

    void destroy_cusparse_matrix(
            cusparseSpMatDescr_t& cusparse_matrix)
    {
        cusparseStatus_t status = cusparseDestroySpMat(cusparse_matrix);
        assert(status == CUSPARSE_STATUS_SUCCESS);
    }


    // =======================
    // destroy cusparse vector
    // =======================

    /// \brief Destroys cusparse vector.
    ///

    void destroy_cusparse_vector(
            cusparseDnVecDescr_t& cusparse_vector)
    {
        cusparseStatus_t status = cusparseDestroyDnVec(cusparse_vector);
        assert(status == CUSPARSE_STATUS_SUCCESS);
    }


    // ===========================
    // cusparse matrix buffer size (float)
    // ===========================

    /// \brief A template wrapper for \cu cusparseSpMat_buffersize for \c float
    ///        precision data. This function determines the buffer size needed
    ///        for matrix-vector multiplication using \c cusparseSpMV. The
    ///        output is \c buffer_size variable.

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


    // ===========================
    // cusparse matrix buffer size (double)
    // ===========================

    /// \brief A template wrapper for \cu cusparseSpMat_buffersize for
    ///        \c double precision data. This function determines the buffer
    ///        size needed for matrix-vector multiplication using
    ///        \c cusparseSpMV. The output is \c buffer_size variable.

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


    // ===============
    // cusparse matvec (float)
    // ===============

    /// \brief A wrapper for \c cusparseSpMV to perform sparse matrix-vector
    ///        multiplication uasing \c float precision data.

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


    // ===============
    // cusparse matvec (double)
    // ===============

    /// \brief A wrapper for \c cusparseSpMV to perform sparse matrix-vector
    ///        multiplication uasing \c double precision data.

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
}  // namespace cusparse_interface
