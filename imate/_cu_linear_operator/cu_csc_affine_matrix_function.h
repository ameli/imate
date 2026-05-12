/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */


#ifndef _CU_LINEAR_OPERATOR_CU_CSC_AFFINE_MATRIX_FUNCTION_H_
#define _CU_LINEAR_OPERATOR_CU_CSC_AFFINE_MATRIX_FUNCTION_H_


// =======
// Headers
// =======

#include "../_definitions/types.h"  // LongIndexType
#include "./cu_csc_matrix.h"  // cuCSCMatrix
#include "./cu_affine_matrix_function.h"  // cuAffineMatrixFunction


// =============================
// cu CSC Affine Matrix Function
// =============================

/// \class   cuCSCAffineMatrixFunction
///
/// \brief   Container for CSC affine matrix functions of one parameter.
///
/// \details The \c cuCSCAffineMatrixFunction contains two-dimensional
///          compressed sparse column matrices \c A and \c B.
///          This operoator can perofrom matrix-vector product and transposed
///          matrix-vector product.
///
/// \sa      cuAffineMatrixFunction,
///          cuDenseMatrixFunction,
///          cuCSRMatrixFunction,
///          cuCSCMatrix,
///          cCSCAffineMatrixFunction

template <typename DataType>
class cuCSCAffineMatrixFunction : public cuAffineMatrixFunction<DataType>
{
    public:

        // Member methods
        cuCSCAffineMatrixFunction(
                const DataType* A_data_,
                const LongIndexType* A_indices_,
                const LongIndexType* A_index_pointer_,
                const LongIndexType num_rows_,
                const LongIndexType num_columns_,
                const FlagType A_is_symmetric_,
                const int num_gpu_devices_);

        cuCSCAffineMatrixFunction(
                const DataType* A_data_,
                const LongIndexType* A_indices_,
                const LongIndexType* A_index_pointer_,
                const LongIndexType num_rows_,
                const LongIndexType num_columns_,
                const FlagType A_is_symmetric_,
                const DataType* B_data_,
                const LongIndexType* B_indices_,
                const LongIndexType* B_index_pointer_,
                const FlagType B_is_symmetric_,
                const int num_gpu_devices_);

        virtual ~cuCSCAffineMatrixFunction();
        
        virtual void set_symmetry(const FlagType symmetric);

        virtual void dot(
                const DataType* vector,
                DataType* product);

        virtual void transpose_dot(
                const DataType* vector,
                DataType* product);

    protected:

        // Member data
        cuCSCMatrix<DataType> A;
        cuCSCMatrix<DataType> B;
};

#endif  // _CU_LINEAR_OPERATOR_CU_CSC_AFFINE_MATRIX_FUNCTION_H_
