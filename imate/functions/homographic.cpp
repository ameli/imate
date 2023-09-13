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

#include "./homographic.h"


// ===========
// Homographic
// ===========

/// \brief Sets the default for the parameter \c a, \c b, \c c, and \c d.
///

Homographic::Homographic(double a_, double b_, double c_, double d_)
{
    this->a = a_;
    this->b = b_;
    this->c = c_;
    this->d = d_;
}


// ====================
// Homographic function (float)
// ====================

/// \param[in] lambda_
///            Eigenvalue (or singular value) of matrix.
/// \return    The value of matrix function for the given eigenvalue.

float Homographic::function(const float lambda_) const
{
    // Casting
    float a_ = static_cast<float>(this->a);
    float b_ = static_cast<float>(this->b);
    float c_ = static_cast<float>(this->c);
    float d_ = static_cast<float>(this->d);

    return (a_ * lambda_ + b_) / (c_ * lambda_ + d_);
}


// ====================
// Homographic function (double)
// ====================

/// \param[in] lambda_
///            Eigenvalue (or singular value) of matrix.
/// \return    The value of matrix function for the given eigenvalue.

double Homographic::function(const double lambda_) const
{
    // Casting
    double a_ = static_cast<double>(this->a);
    double b_ = static_cast<double>(this->b);
    double c_ = static_cast<double>(this->c);
    double d_ = static_cast<double>(this->d);

    return (a_ * lambda_ + b_) / (c_ * lambda_ + d_);
}


// ====================
// Homographic function (long double)
// ====================

/// \param[in] lambda_
///            Eigenvalue (or singular value) of matrix.
/// \return    The value of matrix function for the given eigenvalue.

long double Homographic::function(const long double lambda_) const
{
    // Casting
    long double a_ = static_cast<long double>(this->a);
    long double b_ = static_cast<long double>(this->b);
    long double c_ = static_cast<long double>(this->c);
    long double d_ = static_cast<long double>(this->d);

    return (a_ * lambda_ + b_) / (c_ * lambda_ + d_);
}
