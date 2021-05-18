/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */


#ifndef _C_LINEAR_OPERATOR_C_CSR_AFFINE_MATRIX_FUNCTION_H_
#define _C_LINEAR_OPERATOR_C_CSR_AFFINE_MATRIX_FUNCTION_H_


// =======
// Headers
// =======

#include "../_definitions/types.h"  // FlagType, LongIndexType
#include "./c_affine_matrix_function.h"  // cAffineMatrixFunction
#include "./c_csr_matrix.h"  // cCSRMatrix


// ============================
// c CSR Affine Matrix Function
// ============================

template <typename DataType>
class cCSRAffineMatrixFunction : public cAffineMatrixFunction<DataType>
{
    public:

        // Member methods
        cCSRAffineMatrixFunction(
                const DataType* A_data_,
                const LongIndexType* A_indices_,
                const LongIndexType* A_index_pointer_,
                const LongIndexType num_rows_,
                const LongIndexType num_columns_);

        cCSRAffineMatrixFunction(
                const DataType* A_data_,
                const LongIndexType* A_indices_,
                const LongIndexType* A_index_pointer_,
                const LongIndexType num_rows_,
                const LongIndexType num_colums_,
                const DataType* B_data_,
                const LongIndexType* B_indices_,
                const LongIndexType* B_index_pointer_);

        virtual ~cCSRAffineMatrixFunction();

        virtual void dot(
                const DataType* vector,
                DataType* product);

        virtual void transpose_dot(
                const DataType* vector,
                DataType* product);

    protected:

        // Member data
        cCSRMatrix<DataType> A;
        cCSRMatrix<DataType> B;
};

#endif  // _C_LINEAR_OPERATOR_C_CSR_AFFINE_MATRIX_FUNCTION_H_
