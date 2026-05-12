/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */


#ifndef _C_LINEAR_OPERATOR_C_LINEAR_OPERATOR_H_
#define _C_LINEAR_OPERATOR_C_LINEAR_OPERATOR_H_

// =======
// Headers
// =======

#include "../_definitions/types.h"  // FlagType, IndexType, LongIndexType
#include "./c_linear_operator_base.h"  // cLinearOperatorBase


// =================
// c Linear Operator
// =================

/// \class   cLinearOperator
///
/// \brief   Base class for linear operators. This class serves as interface
///          for all derived classes.
///
/// \details The prefix \c c in this class's name (and its derivatves), stands
///          for the \c cpp code, intrast to the \c cu prefix, which stands for
///          the cuda code. Most derived classes have a cuda counterpart.
///
/// \sa      cMatrix,
///          cAffineMatrixFunction,
///          cuLinearOperator,
///          cLinearOperatorBase

template <typename DataType>
class cLinearOperator : virtual public cLinearOperatorBase
{
    public:

        // Member methods
        cLinearOperator();

        virtual ~cLinearOperator();

        void set_parameters(DataType* parameters_);

        virtual DataType get_eigenvalue(
                const DataType* known_parameters,
                const DataType known_eigenvalue,
                const DataType* inquiry_parameters) const = 0;

        virtual void dot(
                const DataType* vector,
                DataType* product) = 0;

        virtual void transpose_dot(
                const DataType* vector,
                DataType* product) = 0;

    protected:

        // Member data
        DataType* parameters;
};

#endif  // _C_LINEAR_OPERATOR_C_LINEAR_OPERATOR_H_
