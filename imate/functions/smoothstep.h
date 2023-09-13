/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */


#ifndef FUNCTIONS_SMOOTHSTEP_H_
#define FUNCTIONS_SMOOTHSTEP_H_

// =======
// Headers
// =======

#include "./functions.h"


// ===========
// Smooth Step
// ===========

/// \brief   Defines the function
///          \f[
///              f: \lambda \mapsto \frac{1}{2}
///              \left( 1 + \mathrm{tanh}(\alpha \lambda) \right)
///          \f]
///          where \f$ \alpha \f$ is a scale parameter and should be set by
///          \c this->alpha member.
///
/// \details The matrix function
///          \f$ f: \mathbb{R}^{n \times n} \to \mathbb{R}^{n \times n} \f$
///          is used in
///
///          \f[
///              \mathrm{trace} \left( f(\mathbf{A}) \right).
///          \f]
///
///          However, instead of a matrix function, the equivalent scalar
///          function \f$ f: \mathbb{R} \to \mathbb{R} \f$ is defiend which
///          acts on the eigenvalues of the matrix.
///
/// \note    The smooth step function defined here should not be confused with
///          a conventionally used function of the same name using cubic
///          polynomial.

class SmoothStep : public Function
{
    public:
        SmoothStep(double alpha_);
        virtual float function(const float lambda_) const;
        virtual double function(const double lambda_) const;
        virtual long double function(const long double lambda_) const;
        double alpha;
};

#endif  // FUNCTIONS_SMOOTHSTEP_H_
