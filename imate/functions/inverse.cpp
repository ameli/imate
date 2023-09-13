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

#include "./inverse.h"


// ================
// Inverse function (float)
// ================

/// \param[in] lambda_
///            Eigenvalue (or singular value) of matrix.
/// \return    The value of matrix function for the given eigenvalue.

float Inverse::function(const float lambda_) const
{
    return 1.0 / lambda_;
}


// ================
// Inverse function (double)
// ================

/// \param[in] lambda_
///            Eigenvalue (or singular value) of matrix.
/// \return    The value of matrix function for the given eigenvalue.

double Inverse::function(const double lambda_) const
{
    return 1.0 / lambda_;
}


// ================
// Inverse function (long double)
// ================

/// \param[in] lambda_
///            Eigenvalue (or singular value) of matrix.
/// \return    The value of matrix function for the given eigenvalue.

long double Inverse::function(const long double lambda_) const
{
    return 1.0 / lambda_;
}
