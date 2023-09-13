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

#include <cmath>  // exp
#include "./exponential.h"


// ===========
// Exponential
// ===========

/// \brief Sets the default for the parameter \c coeff to \c 1.0.
///

Exponential::Exponential(double coeff_)
{
    this->coeff = coeff_;
}


// ====================
// Exponential function (float)
// ====================

/// \param[in] lambda_
///            Eigenvalue (or singular value) of matrix.
/// \return    The value of matrix function for the given eigenvalue.

float Exponential::function(const float lambda_) const
{
    return exp(lambda_ * static_cast<float>(this->coeff));
}


// ====================
// Exponential function (double)
// ====================

/// \param[in] lambda_
///            Eigenvalue (or singular value) of matrix.
/// \return    The value of matrix function for the given eigenvalue.

double Exponential::function(const double lambda_) const
{
    return exp(lambda_ * static_cast<double>(this->coeff));
}


// ====================
// Exponential function (long double)
// ====================

/// \param[in] lambda_
///            Eigenvalue (or singular value) of matrix.
/// \return    The value of matrix function for the given eigenvalue.

long double Exponential::function(const long double lambda_) const
{
    return exp(lambda_ * static_cast<long double>(this->coeff));
}
