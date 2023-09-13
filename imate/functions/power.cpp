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

#include <cmath>  // pow
#include "./power.h"


// =====
// Power
// =====

/// \brief Sets the default for the parameter \c exponent to \c 2.0.
///

Power::Power():
    exponent(2.0)
{
}


// ==============
// Power function (float)
// ==============

/// \param[in] lambda_
///            Eigenvalue (or singular value) of matrix.
/// \return    The value of matrix function for the given eigenvalue.

float Power::function(const float lambda_) const
{
    return pow(lambda_, static_cast<float>(this->exponent));
}


// ==============
// Power function (double)
// ==============

/// \param[in] lambda_
///            Eigenvalue (or singular value) of matrix.
/// \return    The value of matrix function for the given eigenvalue.

double Power::function(const double lambda_) const
{
    return pow(lambda_, this->exponent);
}


// ==============
// Power function (long double)
// ==============

/// \param[in] lambda_
///            Eigenvalue (or singular value) of matrix.
/// \return    The value of matrix function for the given eigenvalue.

long double Power::function(const long double lambda_) const
{
    return pow(lambda_, static_cast<long double>(this->exponent));
}
