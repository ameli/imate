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

#include <cmath>  // tanh
#include "./smoothstep.h"


// ===========
// Smooth Step
// ===========

/// \brief Sets the default for the parameter \c alpha to \c 1.0.
///

SmoothStep::SmoothStep(double alpha_)
{
    this->alpha = alpha_;
}


// ====================
// Smooth Step function (float)
// ====================

/// \param[in] lambda_
///            Eigenvalue (or singular value) of matrix.
/// \return    The value of matrix function for the given eigenvalue.

float SmoothStep::function(const float lambda_) const
{
    return 0.5 * (1.0 + tanh(static_cast<float>(this->alpha) * lambda_));
}


// ====================
// Smooth Step function (double)
// ====================

/// \param[in] lambda_
///            Eigenvalue (or singular value) of matrix.
/// \return    The value of matrix function for the given eigenvalue.

double SmoothStep::function(const double lambda_) const
{
    return 0.5 * (1.0 + tanh(this->alpha * lambda_));
}


// ====================
// Smooth Step function (long double)
// ====================

/// \param[in] lambda_
///            Eigenvalue (or singular value) of matrix.
/// \return    The value of matrix function for the given eigenvalue.

long double SmoothStep::function(const long double lambda_) const
{
    return 0.5 * (1.0 + tanh(static_cast<long double>(this->alpha) * lambda_));
}
