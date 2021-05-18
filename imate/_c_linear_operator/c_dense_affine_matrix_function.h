/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */


#ifndef _C_LINEAR_OPERATOR_C_DENSE_AFFINE_MATRIX_FUNCTION_H_
#define _C_LINEAR_OPERATOR_C_DENSE_AFFINE_MATRIX_FUNCTION_H_


// =======
// Headers
// =======

#include "../_definitions/types.h"  // LongIndexType, FlagType
#include "./c_affine_matrix_function.h"  // cAffineMatrixFunction
#include "./c_dense_matrix.h"  // cDenseMatrix


// ==============================
// c Dense Affine Matrix Function
// ==============================

template <typename DataType>
class cDenseAffineMatrixFunction : public cAffineMatrixFunction<DataType>
{
    public:

        // Member methods
        cDenseAffineMatrixFunction(
                const DataType* A_,
                const FlagType A_is_row_major_,
                const LongIndexType num_rows_,
                const LongIndexType num_colums_);

        cDenseAffineMatrixFunction(
                const DataType* A_,
                const FlagType A_is_row_major_,
                const LongIndexType num_rows_,
                const LongIndexType num_columns_,
                const DataType* B_,
                const FlagType B_is_row_major_);

        virtual ~cDenseAffineMatrixFunction();

        virtual void dot(
                const DataType* vector,
                DataType* product);

        virtual void transpose_dot(
                const DataType* vector,
                DataType* product);

    protected:

        // Member data
        cDenseMatrix<DataType> A;
        cDenseMatrix<DataType> B;
};

#endif  // _C_LINEAR_OPERATOR_C_DENSE_AFFINE_MATRIX_FUNCTION_H_
