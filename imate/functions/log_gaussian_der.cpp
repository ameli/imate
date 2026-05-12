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

// Before including cmath, define _USE_MATH_DEFINES. This is only required to
// define the math constants like M_PI, etc, in win32 operating system.
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32) && \
    !defined(__CYGWIN__)
    #define _USE_MATH_DEFINES
#endif

#include <cmath>  // exp, M_SQRT1_2, M_2_SQRTPI
#include "./log_gaussian_der.h"


// ================
// Log Gaussian Der
// ================

/// \brief Sets the default for the parameter \c mu to \c 0.0 and for the
///        parameter \c sigma to \c 1.0.

LogGaussianDer::LogGaussianDer(double mu_, double sigma_)
{
    this->mu = mu_;
    this->sigma = sigma_;
}

// =========================
// Log Gaussian Der function (float)
// =========================

/// \param[in] lambda_
///            Eigenvalue (or singular value) of matrix.
/// \return    The value of matrix function for the given eigenvalue.

float LogGaussianDer::function(const float lambda_) const
{
    float mu_ = static_cast<float>(this->mu);
    float sigma_ = static_cast<float>(this->sigma);
    float sigma2 = sigma_ * sigma_;
    float sigma3 = sigma2 * sigma_;
    float x = (lambda_ - mu_) / sigma_;
    return -(0.5f * M_SQRT1_2 * M_2_SQRTPI / sigma3) * \
        std::exp(-0.5f * x * x) * \
        (std::log(lambda_) - std::log(mu) + sigma2) / (lambda_ * lambda_);
}


// =========================
// Log Gaussian Der function (double)
// =========================

/// \param[in] lambda_
///            Eigenvalue (or singular value) of matrix.
/// \return    The value of matrix function for the given eigenvalue.

double LogGaussianDer::function(const double lambda_) const
{
    double x = (lambda_ - this->mu) / this->sigma;
    double sigma2 = this->sigma * this->sigma;
    double sigma3 = sigma2 * this->sigma;
    return -(0.5 * M_SQRT1_2 * M_2_SQRTPI / sigma3) * \
        std::exp(-0.5 * x * x) * \
        (std::log(lambda_) - std::log(mu) + sigma2) / (lambda_ * lambda_);
}


// =========================
// Log Gaussian Der function (long double)
// =========================

/// \param[in] lambda_
///            Eigenvalue (or singular value) of matrix.
/// \return    The value of matrix function for the given eigenvalue.

long double LogGaussianDer::function(const long double lambda_) const
{
    long double mu_ = static_cast<long double>(this->mu);
    long double sigma_ = static_cast<long double>(this->sigma);
    long double sigma2 = sigma_ * sigma_;
    long double sigma3 = sigma2 * sigma_;
    long double x = (lambda_ - mu_) / sigma_;
    return -(0.5l * M_SQRT1_2 * M_2_SQRTPI / sigma3) * \
        std::exp(-0.5l * x * x) * \
        (std::log(lambda_) - std::log(mu) + sigma2) / (lambda_ * lambda_);
}
