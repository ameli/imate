/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */


#ifndef _C_LINEAR_OPERATOR_C_CSC_AFFINE_MATRIX_FUNCTION_H_
#define _C_LINEAR_OPERATOR_C_CSC_AFFINE_MATRIX_FUNCTION_H_


// =======
// Headers
// =======

#include "../_definitions/types.h"  // LongIndexType
#include "./c_affine_matrix_function.h"  // cAffineMatrixFunction
#include "./c_csc_matrix.h"  // cCSCMatrix


// ============================
// c CSC Affine Matrix Function
// ============================

/// \class   cCSCAffineMatrixFunction
///
/// \brief   Container for CSC affine matrix functions of one parameter.
///
/// \details The \c cCSCAffineMatrixFunction contains two-dimensional
///          compressed sparse column matrices \c A and \c B.
///          This operoator can perofrom matrix-vector product and transposed
///          matrix-vector product.
///
/// \sa      cAffineMatrixFunction,
///          cDenseMatrixFunction,
///          cCSRMatrixFunction,
///          cCSCMatrix,
///          cuCSCAffineMatrixFunction

template <typename DataType>
class cCSCAffineMatrixFunction : public cAffineMatrixFunction<DataType>
{
    public:

        // Member methods
        cCSCAffineMatrixFunction(
                const DataType* A_data_,
                const LongIndexType* A_indices_,
                const LongIndexType* A_index_pointer_,
                const LongIndexType num_rows_,
                const LongIndexType num_columns_,
                const FlagType A_is_symmetric_);

        cCSCAffineMatrixFunction(
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

        virtual ~cCSCAffineMatrixFunction();
        
        virtual void set_symmetry(const FlagType symmetric);

        virtual void dot(
                const DataType* vector,
                DataType* product);

        virtual void transpose_dot(
                const DataType* vector,
                DataType* product);

    protected:

        // Member data
        cCSCMatrix<DataType> A;
        cCSCMatrix<DataType> B;
};

#endif  // _C_LINEAR_OPERATOR_C_CSC_AFFINE_MATRIX_FUNCTION_H_
