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

#include "./c_linear_operator.h"
#include <cstddef>  // NULL


// =============
// constructor 1
// =============

/// \brief Default constructor.
///

template <typename DataType>
cLinearOperator<DataType>::cLinearOperator():
    num_rows(0),
    num_columns(0),
    eigenvalue_relation_known(0),
    parameters(NULL),
    num_parameters(0)
{
}


// =============
// constructor 2
// =============

/// \brief  Constructor with setting \c num_rows and \c num_columns.
///
/// \note   For the classed that are virtually derived (virtual inheritance)
///         from this class, this constructor will never be called. Rather, the
///         default constructor is called by the most derived class. Thus, set
///         the member data directly instead of below.

template <typename DataType>
cLinearOperator<DataType>::cLinearOperator(
        const LongIndexType num_rows_,
        const LongIndexType num_columns_):

    // Initializer list
    num_rows(num_rows_),
    num_columns(num_columns_),
    eigenvalue_relation_known(0),
    parameters(NULL),
    num_parameters(0)
{
}


// ==========
// destructor
// ==========

template <typename DataType>
cLinearOperator<DataType>::~cLinearOperator()
{
}


// ============
// get num rows
// ============

template <typename DataType>
LongIndexType cLinearOperator<DataType>::get_num_rows() const
{
    return this->num_rows;
}


// ===============
// get num columns
// ===============

template <typename DataType>
LongIndexType cLinearOperator<DataType>::get_num_columns() const
{
    return this->num_columns;
}


// ==============
// set parameters
// ==============

/// \brief     Sets the scalar parameter \c this->parameters. Parameter is
///            initialized to \c NULL. However, before calling \c dot or
///            \c transpose_dot functions, the parameters must be set.
///
/// \param[in] parameters_
///            A pointer to the scalar or array of parameters.

template <typename DataType>
void cLinearOperator<DataType>::set_parameters(DataType* parameters_)
{
    this->parameters = parameters_;
}


// ==================
// get num parameters
// ==================

template <typename DataType>
IndexType cLinearOperator<DataType>::get_num_parameters() const
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

template <typename DataType>
FlagType cLinearOperator<DataType>::is_eigenvalue_relation_known() const
{
    return this->eigenvalue_relation_known;
}


// ===============================
// Explicit template instantiation
// ===============================

template class cLinearOperator<float>;
template class cLinearOperator<double>;
template class cLinearOperator<long double>;
