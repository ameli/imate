/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */


#ifndef FUNCTIONS_GAUSSIAN_H_
#define FUNCTIONS_GAUSSIAN_H_

// =======
// Headers
// =======

#include "./functions.h"


// ========
// Gaussian
// ========

/// \brief   Defines the function
///          \f[
///              f: \lambda \mapsto \frac{1}{\sigma \sqrt{2 \pi}}
///              e^{-\frac{1}{2} \frac{(\lambda - \mu)^2}{\sigma^2}},
///          \f]
///          where \f$ \mu \f$ and \f$ \sigma \f$ parameters are the mean and
///          standard deviation of the Gaussian function and should be set by
///          \c this->mu and \c this->sigma members, respectively.
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

class Gaussian : public Function
{
    public:
        Gaussian(double mu_, double sigma_);
        virtual float function(const float lambda_) const;
        virtual double function(const double lambda_) const;
        virtual long double function(const long double lambda_) const;
        double mu;
        double sigma;
};

#endif  // FUNCTIONS_GAUSSIAN_H_
