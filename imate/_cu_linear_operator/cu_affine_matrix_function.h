/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */


#ifndef _CU_LINEAR_OPERATOR_CU_AFFINE_MATRIX_FUNCTION_H_
#define _CU_LINEAR_OPERATOR_CU_AFFINE_MATRIX_FUNCTION_H_


// =======
// Headers
// =======

#include "../_definitions/types.h"  // LongIndexType
#include "./cu_linear_operator.h"  // cuLinearOperator


// =========================
// cu Affine Matrix Function
// =========================

/// \class   cuAffineMatrixFunction
///
/// \brief   Base class for affine matrix functions of one parameter.
///
/// \details The prefix \c c in this class's name (and its derivatves), stands
///          for the \c cpp code, intrast to the \c cu prefix, which stands for
///          the cuda code. Most derived classes have a cuda counterpart.
///
/// \sa      cuMatrix


template <typename DataType>
class cuAffineMatrixFunction : public cuLinearOperator<DataType>
{
    public:

        // Member methods
        cuAffineMatrixFunction();

        virtual ~cuAffineMatrixFunction();

        void set_parameters(DataType* t);

        DataType get_eigenvalue(
                const DataType* known_parameters,
                const DataType known_eigenvalue,
                const DataType* inquiry_parameters) const;

    protected:

        // Member methods
        void _add_scaled_vector(
                const DataType* input_vector,
                const LongIndexType vector_size,
                const DataType scale,
                DataType* output_vector) const;

        // Member data
        bool B_is_identity;
};


#endif  // _CU_LINEAR_OPERATOR_CU_AFFINE_MATRIX_FUNCTION_H_
