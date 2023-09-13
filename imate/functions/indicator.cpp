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

#include "./indicator.h"


// =========
// Indicator
// =========

/// \brief  Sets the default parameters \c a and \b c.
///

Indicator::Indicator(double a_, double b_)
{
    this->a = a_;
    this->b = b_;
}


// ==================
// Indicator function (float)
// ==================

/// \param[in] lambda_
///            Eigenvalue (or singular value) of matrix.
/// \return    The value of matrix function for the given eigenvalue.

float Indicator::function(const float lambda_) const
{
    if ((lambda_ < this->a) || (lambda_ > this->b))
    {
        return 0.0;
    }
    else
    {
        return 1.0;
    }
}


// ==================
// Indicator function (double)
// ==================

/// \param[in] lambda_
///            Eigenvalue (or singular value) of matrix.
/// \return    The value of matrix function for the given eigenvalue.

double Indicator::function(const double lambda_) const
{
    if ((lambda_ < this->a) || (lambda_ > this->b))
    {
        return 0.0;
    }
    else
    {
        return 1.0;
    }
}


// ==================
// Indicator function (long double)
// ==================

/// \param[in] lambda_
///            Eigenvalue (or singular value) of matrix.
/// \return    The value of matrix function for the given eigenvalue.

long double Indicator::function(const long double lambda_) const
{
    if ((lambda_ < this->a) || (lambda_ > this->b))
    {
        return 0.0;
    }
    else
    {
        return 1.0;
    }
}
