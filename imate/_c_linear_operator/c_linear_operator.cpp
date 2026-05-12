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
    parameters(NULL)
{
}


// ==========
// destructor
// ==========

/// \brief Destructor.
///

template <typename DataType>
cLinearOperator<DataType>::~cLinearOperator()
{
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


// ===============================
// Explicit template instantiation
// ===============================

template class cLinearOperator<float>;
template class cLinearOperator<double>;
template class cLinearOperator<long double>;
