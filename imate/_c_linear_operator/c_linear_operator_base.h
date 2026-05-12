/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */


#ifndef _C_LINEAR_OPERATOR_C_LINEAR_OPERATOR_BASE_H_
#define _C_LINEAR_OPERATOR_C_LINEAR_OPERATOR_BASE_H_

// =======
// Headers
// =======

#include "../_definitions/types.h"  // FlagType, IndexType, LongIndexType


// ======================
// c Linear Operator Base
// ======================

/// \class   cLinearOperatorBase
///
/// \brief   Base class for \c cLinearOperator and \c cuLinearOperator . This
///          class is not templated so that both cpp and cu classed can be
///          derived from it without conflict of data types.
///
/// \details The prefix \c c in this class's name (and its derivatves), stands
///          for the \c cpp code, intrast to the \c cu prefix, which stands for
///          the cuda code. Most derived classes have a cuda counterpart.
///
/// \sa      cLinearOperator,
///          cuLinearOperator

class cLinearOperatorBase
{
    public:

        // Member methods
        cLinearOperatorBase();

        cLinearOperatorBase(
                const LongIndexType num_rows_,
                const LongIndexType num_columns_);

        virtual ~cLinearOperatorBase();

        LongIndexType get_num_rows() const;
        LongIndexType get_num_columns() const;
        IndexType get_num_parameters() const;
        FlagType is_eigenvalue_relation_known() const;

        virtual void set_symmetry(FlagType symmetric) = 0;

    protected:

        // Member data
        const LongIndexType num_rows;
        const LongIndexType num_columns;
        FlagType eigenvalue_relation_known;
        IndexType num_parameters;
};

#endif  // _C_LINEAR_OPERATOR_C_LINEAR_OPERATOR_BASE_H_
