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
///          cuLinearOperator

template <typename DataType>
class cLinearOperator
{
    public:

        // Member methods
        cLinearOperator();

        cLinearOperator(
                const LongIndexType num_rows_,
                const LongIndexType num_columns_);

        virtual ~cLinearOperator();

        LongIndexType get_num_rows() const;
        LongIndexType get_num_columns() const;
        void set_parameters(DataType* parameters_);
        IndexType get_num_parameters() const;
        FlagType is_eigenvalue_relation_known() const;

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
        const LongIndexType num_rows;
        const LongIndexType num_columns;
        FlagType eigenvalue_relation_known;
        DataType* parameters;
        IndexType num_parameters;
};

#endif  // _C_LINEAR_OPERATOR_C_LINEAR_OPERATOR_H_
