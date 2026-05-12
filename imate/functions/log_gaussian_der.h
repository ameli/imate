/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */


#ifndef FUNCTIONS_LOG_GAUSSIAN_DER_H_
#define FUNCTIONS_LOG_GAUSSIAN_DER_H_

// =======
// Headers
// =======

#include "./functions.h"


// ================
// Log Gaussian Der
// ================

/// \brief   Defines the derivative of the log normal probability function by
///          \f[
///              f: \lambda \mapsto -\frac{1}{\sigma^3 \sqrt{2 \pi}}
///              e^{-\frac{1}{2} x^2} 
///              \frac{\log(\lambda) - \log(\mu) + \sigma^2}{\lambda^2},
///          \f]
///          where
///          \f[
///              x = \frac{\lambda - \mu}{\sigma},
///          \f]
///          and \f$ \mu \f$ and \f$ \sigma \f$ parameters are the mean and
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

class LogGaussianDer : public Function
{
    public:
        LogGaussianDer(double mu_, double sigma_);
        virtual float function(const float lambda_) const;
        virtual double function(const double lambda_) const;
        virtual long double function(const long double lambda_) const;
        double mu;
        double sigma;
};

#endif  // FUNCTIONS_LOG_GAUSSIAN_DER_H_
