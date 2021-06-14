/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */


#ifndef _CU_LINEAR_OPERATOR_CU_CSR_MATRIX_H_
#define _CU_LINEAR_OPERATOR_CU_CSR_MATRIX_H_


// =======
// Headers
// =======

#include "../_definitions/types.h"  // FlagType, LongIndexType
#include "../_c_linear_operator/c_csr_matrix.h"  // cCSRMatrix
#include "./cu_matrix.h"  // cuMatrix


// =============
// cu CSR Matrix
// =============

template <typename DataType>
class cuCSRMatrix :
    public cuMatrix<DataType>,
    public cCSRMatrix<DataType>
{
    public:

        // Member methods
        cuCSRMatrix();

        cuCSRMatrix(
                const DataType* A_data_,
                const LongIndexType* A_indices_,
                const LongIndexType* A_index_pointer_,
                const LongIndexType num_rows_,
                const LongIndexType num_columns_,
                const int num_gpu_devices_);

        virtual ~cuCSRMatrix();

        virtual void dot(
                const DataType* device_vector,
                DataType* device_product);

        virtual void dot_plus(
                const DataType* device_vector,
                const DataType alpha,
                DataType* device_product);

        virtual void transpose_dot(
                const DataType* device_vector,
                DataType* device_product);

        virtual void transpose_dot_plus(
                const DataType* device_vector,
                const DataType alpha,
                DataType* device_product);

    protected:

        // Member methods
        virtual void copy_host_to_device();

        void allocate_buffer(
                const int device_id,
                cusparseOperation_t cusparse_operation,
                const DataType alpha,
                const DataType beta,
                cusparseDnVecDescr_t& cusparse_input_vector,
                cusparseDnVecDescr_t& cusparse_output_vector,
                cusparseSpMVAlg_t algorithm);

        // Member data
        DataType** device_A_data;
        LongIndexType** device_A_indices;
        LongIndexType** device_A_index_pointer;
        void** device_buffer;
        size_t* device_buffer_num_bytes;
        cusparseSpMatDescr_t* cusparse_matrix_A;
};

#endif  // _CU_LINEAR_OPERATOR_CU_CSR_MATRIX_H_
