/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */


// =======
// Headers
// =======

#include "./c_linear_operator_base.h"
#include <cstddef>  // NULL


// =============
// constructor 1
// =============

/// \brief Default constructor.
///

cLinearOperatorBase::cLinearOperatorBase():
    num_rows(0),
    num_columns(0),
    eigenvalue_relation_known(0),
    num_parameters(0)
{
}


// =============
// constructor 2
// =============

/// \brief  Constructor with setting \c num_rows and \c num_columns.
///
/// \note       For the classed that are virtually derived (virtual
///             inheritance) from this class, this constructor will never be
///             called. Rather, the default constructor is called by the most
///             derived class. Thus, set the member data directly instead of
///             below.
///
/// \param[in]  num_rows_
///             Number of rows of \c A
/// \param[in]  num_columns_
///             Number of columns of \c A

cLinearOperatorBase::cLinearOperatorBase(
        const LongIndexType num_rows_,
        const LongIndexType num_columns_):

    // Initializer list
    num_rows(num_rows_),
    num_columns(num_columns_),
    eigenvalue_relation_known(0),
    num_parameters(0)
{
}


// ==========
// destructor
// ==========

/// \brief Destructor.
///

cLinearOperatorBase::~cLinearOperatorBase()
{
}


// ============
// get num rows
// ============

/// \brief  Returns the number of rows of the matrix.
///
/// \return Number of matrix rows

LongIndexType cLinearOperatorBase::get_num_rows() const
{
    return this->num_rows;
}


// ===============
// get num columns
// ===============

/// \brief  Returns the number of columns of the matrix.
///
/// \return Number of matrix columns

LongIndexType cLinearOperatorBase::get_num_columns() const
{
    return this->num_columns;
}


// ==================
// get num parameters
// ==================

/// \brief   Returns the number of parameters of the linear operator.
///
/// \details For the subclass \c cMatrix, this value is zero. For the subclass
///          \c cAffineMatrixFunction, this value is a non-zero integer.
///
/// \return  Number of nonzero elements

IndexType cLinearOperatorBase::get_num_parameters() const
{
    return this->num_parameters;
}


// ============================
// is eigenvalue relation known
// ============================

/// \brief  Returns a flag that determines whether a relation between the
///         parameters of the operator and its eigenvalue(s) is known.
///
/// \return If the relation between parameters and eigenvalue of the
///         operator is known, returns \c 1, otherwise returns \c 0.

FlagType cLinearOperatorBase::is_eigenvalue_relation_known() const
{
    return this->eigenvalue_relation_known;
}
