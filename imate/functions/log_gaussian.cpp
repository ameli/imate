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

#include <cmath>  // exp, log, M_SQRT1_2, M_2_SQRTPI
#include "./log_gaussian.h"


// ============
// Log Gaussian
// ============

/// \brief Sets the default for the parameter \c mu to \c 0.0 and for the
///        parameter \c sigma to \c 1.0.

LogGaussian::LogGaussian(double mu_, double sigma_)
{
    this->mu = mu_;
    this->sigma = sigma_;
}

// =====================
// Log Gaussian function (float)
// =====================

/// \param[in] lambda_
///            Eigenvalue (or singular value) of matrix.
/// \return    The value of matrix function for the given eigenvalue.

float LogGaussian::function(const float lambda_) const
{
    float mu_ = static_cast<float>(this->mu);
    float sigma_ = static_cast<float>(this->sigma);
    float x = (std::log(lambda_) - std::log(mu_)) / sigma_;
    return (0.5f * M_SQRT1_2 * M_2_SQRTPI / sigma_) * \
        std::exp(-0.5f * x * x) / lambda_;
}


// =====================
// Log Gaussian function (double)
// =====================

/// \param[in] lambda_
///            Eigenvalue (or singular value) of matrix.
/// \return    The value of matrix function for the given eigenvalue.

double LogGaussian::function(const double lambda_) const
{
    double x = (std::log(lambda_) - std::log(this->mu)) / this->sigma;
    return (0.5 * M_SQRT1_2 * M_2_SQRTPI / this->sigma) * \
        std::exp(-0.5 * x * x) / lambda_;
}


// =====================
// Log Gaussian function (long double)
// =====================

/// \param[in] lambda_
///            Eigenvalue (or singular value) of matrix.
/// \return    The value of matrix function for the given eigenvalue.

long double LogGaussian::function(const long double lambda_) const
{
    long double mu_ = static_cast<long double>(this->mu);
    long double sigma_ = static_cast<long double>(this->sigma);
    long double x = (std::log(lambda_) - std::log(mu_)) / sigma_;
    return (0.5l * M_SQRT1_2 * M_2_SQRTPI / sigma_) * \
        std::exp(-0.5l * x * x) / lambda_;
}
