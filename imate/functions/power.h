/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */


#ifndef FUNCTIONS_POWER_H_
#define FUNCTIONS_POWER_H_

// =======
// Headers
// =======

#include "./functions.h"


// =====
// Power
// =====

/// \brief   Defines the function \f$ f: \lambda \mapsto \lambda^{p} \f$,
///          where \f$ p \in \mathbb{R} \f$ is a parameter and should be set by
///          \c this->exponent member.
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

class Power : public Function
{
    public:
        Power();
        virtual float function(const float lambda_) const;
        virtual double function(const double lambda_) const;
        virtual long double function(const long double lambda_) const;
        double exponent;
};

#endif  // FUNCTIONS_POWER_H_
