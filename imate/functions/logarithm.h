/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */


#ifndef FUNCTIONS_LOGARITHM_H_
#define FUNCTIONS_LOGARITHM_H_

// =======
// Headers
// =======

#include "./functions.h"


// =========
// Logarithm
// =========

/// \brief   Defines the function \f$ f: \lambda \mapsto \log(\lambda) \f$.
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

class Logarithm : public Function
{
    public:
        virtual float function(const float lambda_) const;
        virtual double function(const double lambda_) const;
        virtual long double function(const long double lambda_) const;
};

#endif  // FUNCTIONS_LOGARITHM_H_
