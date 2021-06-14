/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */


#ifndef _CU_LINEAR_OPERATOR_CU_CSR_AFFINE_MATRIX_FUNCTION_H_
#define _CU_LINEAR_OPERATOR_CU_CSR_AFFINE_MATRIX_FUNCTION_H_


// =======
// Headers
// =======

#include "../_definitions/types.h"  // FlagType, LongIndexType
#include "./cu_csr_matrix.h"  // cuCSRMatrix
#include "./cu_affine_matrix_function.h"  // cuAffineMatrixFunction


// =============================
// cu CSR Affine Matrix Function
// =============================

template <typename DataType>
class cuCSRAffineMatrixFunction : public cuAffineMatrixFunction<DataType>
{
    public:

        // Member methods
        cuCSRAffineMatrixFunction(
                const DataType* A_data_,
                const LongIndexType* A_indices_,
                const LongIndexType* A_index_pointer_,
                const LongIndexType num_rows_,
                const LongIndexType num_columns_,
                const int num_gpu_devices_);

        cuCSRAffineMatrixFunction(
                const DataType* A_data_,
                const LongIndexType* A_indices_,
                const LongIndexType* A_index_pointer_,
                const LongIndexType num_rows_,
                const LongIndexType num_colums_,
                const DataType* B_data_,
                const LongIndexType* B_indices_,
                const LongIndexType* B_index_pointer_,
                const int num_gpu_devices_);

        virtual ~cuCSRAffineMatrixFunction();

        virtual void dot(
                const DataType* vector,
                DataType* product);

        virtual void transpose_dot(
                const DataType* vector,
                DataType* product);

    protected:

        // Member data
        cuCSRMatrix<DataType> A;
        cuCSRMatrix<DataType> B;
};

#endif  // _CU_LINEAR_OPERATOR_CU_CSR_AFFINE_MATRIX_FUNCTION_H_
