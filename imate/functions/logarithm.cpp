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

#include <cmath>  // log
#include "./logarithm.h"


// ==================
// Logarithm function (float)
// ==================

/// \param[in] lambda_
///            Eigenvalue (or singular value) of matrix.
/// \return    The value of matrix function for the given eigenvalue.

float Logarithm::function(const float lambda_) const
{
    return log(lambda_);
}


// ==================
// Logarithm function (double)
// ==================

/// \param[in] lambda_
///            Eigenvalue (or singular value) of matrix.
/// \return    The value of matrix function for the given eigenvalue.

double Logarithm::function(const double lambda_) const
{
    return log(lambda_);
}


// ==================
// Logarithm function (long double)
// ==================

/// \param[in] lambda_
///            Eigenvalue (or singular value) of matrix.
/// \return    The value of matrix function for the given eigenvalue.

long double Logarithm::function(const long double lambda_) const
{
    return log(lambda_);
}
