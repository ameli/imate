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

/// \class   cCSRAffineMatrixFunction
///
/// \brief   Container for CSR affine matrix functions of one parameter.
///
/// \details The \c cCSRAffineMatrixFunction contains two-dimensional
///          compressed sparse row matrices \c A and \c B.
///          This operoator can perofrom matrix-vector product and transposed
///          matrix-vector product.
///
/// \sa      cAffineMatrixFunction,
///          cDenseMatrixFunction,
///          cCSCMatrixFunction,
///          cCSRMatrix,
///          cuCSRAffineMatrixFunction

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
                const LongIndexType num_columns_,
                const FlagType A_is_symmetric_);

        cCSRAffineMatrixFunction(
                const DataType* A_data_,
                const LongIndexType* A_indices_,
                const LongIndexType* A_index_pointer_,
                const LongIndexType num_rows_,
                const LongIndexType num_columns_,
                const FlagType A_is_symmetric_,
                const DataType* B_data_,
                const LongIndexType* B_indices_,
                const LongIndexType* B_index_pointer_,
                const FlagType B_is_symmetric_);

        virtual ~cCSRAffineMatrixFunction();
        
        virtual void set_symmetry(const FlagType symmetric);

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
