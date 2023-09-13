/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */


#ifndef FUNCTIONS_HOMOGRAPHIC_H_
#define FUNCTIONS_HOMOGRAPHIC_H_

// =======
// Headers
// =======

#include "./functions.h"


// ===========
// Homographic
// ===========

/// \brief   Defines the function \f$ f: \lambda \mapsto \frac{a \lambda + b}{
///          c \lambda + d} \f$.
///
/// \details The matrix function
///          \f$ f: \mathbb{R}^{n \times n} \to \mathbb{R}^{n \times n} \f$ is
///          used in
///
///          \f[
///              \mathrm{trace} \left( f(\mathbf{A}) \right).
///          \f]
///
///          However, instead of a matrix function, the equivalent scalar
///          function \f$ f: \mathbb{R} \to \mathbb{R} \f$ is defiend which
///          acts on the eigenvalues of the matrix.

class Homographic : public Function
{
    public:
        Homographic(double a_, double b_, double c_, double d_);
        virtual float function(const float lambda_) const;
        virtual double function(const double lambda_) const;
        virtual long double function(const long double lambda_) const;
        double a;
        double b;
        double c;
        double d;
};

#endif  // FUNCTIONS_HOMOGRAPHIC_H_
