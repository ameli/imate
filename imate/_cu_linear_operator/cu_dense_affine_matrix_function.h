/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */


#ifndef _CU_LINEAR_OPERATOR_CU_DENSE_AFFINE_MATRIX_FUNCTION_H_
#define _CU_LINEAR_OPERATOR_CU_DENSE_AFFINE_MATRIX_FUNCTION_H_


// =======
// Headers
// =======

#include "../_definitions/types.h"  // LongIndexType, FlagType,
#include "./cu_affine_matrix_function.h"  // cuAffineMatrixFunction
#include "./cu_dense_matrix.h"  // cuDenseMatrix


// ===============================
// cu Dense Affine Matrix Function
// ===============================

template <typename DataType>
class cuDenseAffineMatrixFunction : public cuAffineMatrixFunction<DataType>
{
    public:

        // Member methods
        cuDenseAffineMatrixFunction(
                const DataType* A_,
                const FlagType A_is_row_major_,
                const LongIndexType num_rows_,
                const LongIndexType num_colums_,
                const int num_gpu_devices_);

        cuDenseAffineMatrixFunction(
                const DataType* A_,
                const FlagType A_is_row_major_,
                const LongIndexType num_rows_,
                const LongIndexType num_columns_,
                const DataType* B_,
                const FlagType B_is_row_major_,
                const int num_gpu_devices_);

        virtual ~cuDenseAffineMatrixFunction();

        virtual void dot(
                const DataType* vector,
                DataType* product);

        virtual void transpose_dot(
                const DataType* vector,
                DataType* product);

    protected:

        // Member data
        cuDenseMatrix<DataType> A;
        cuDenseMatrix<DataType> B;
};

#endif  // _CU_LINEAR_OPERATOR_CU_DENSE_AFFINE_MATRIX_FUNCTION_H_
