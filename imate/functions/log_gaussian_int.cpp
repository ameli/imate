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

#include <cmath>  // erf, log, M_SQRT1_2
#include "./log_gaussian_int.h"


// ================
// Log Gaussian Int
// ================

/// \brief Sets the default for the parameter \c mu to \c 0.0 and for the
///        parameter \c sigma to \c 1.0.

LogGaussianInt::LogGaussianInt(double mu_, double sigma_)
{
    this->mu = mu_;
    this->sigma = sigma_;
}

// =========================
// Log Gaussian Int function (float)
// =========================

/// \param[in] lambda_
///            Eigenvalue (or singular value) of matrix.
/// \return    The value of matrix function for the given eigenvalue.

float LogGaussianInt::function(const float lambda_) const
{
    float mu_ = static_cast<float>(this->mu);
    float sigma_ = static_cast<float>(this->sigma);
    float x = (std::log(lambda_) - std::log(mu_)) / sigma_;

    // C++11 and later
    #if __cplusplus >= 201103L
        return 0.5f * (1.0f + std::erf(x * M_SQRT1_2));
    #else
        return 0.5f * (1.0f + static_cast<float>(erf(x * M_SQRT1_2)));
    #endif
}


// =========================
// Log Gaussian Int function (double)
// =========================

/// \param[in] lambda_
///            Eigenvalue (or singular value) of matrix.
/// \return    The value of matrix function for the given eigenvalue.

double LogGaussianInt::function(const double lambda_) const
{
    double x = (std::log(lambda_) - std::log(this->mu)) / this->sigma;

    // C++11 and later
    #if __cplusplus >= 201103L
        return 0.5 * (1.0 + std::erf(x * M_SQRT1_2));
    #else
        return 0.5 * (1.0 + erf(x * M_SQRT1_2));
    #endif
}


// =========================
// Log Gaussian Int function (long double)
// =========================

/// \param[in] lambda_
///            Eigenvalue (or singular value) of matrix.
/// \return    The value of matrix function for the given eigenvalue.

long double LogGaussianInt::function(const long double lambda_) const
{
    long double mu_ = static_cast<long double>(this->mu);
    long double sigma_ = static_cast<long double>(this->sigma);
    long double x = (std::log(lambda_) - std::log(mu_)) / sigma_;

    // C++11 and later
    #if __cplusplus >= 201103L
        return 0.5l * (1.0l + std::erf(x * M_SQRT1_2));
    #else
        return 0.5l * (1.0l + static_cast<long double>(erf(x * M_SQRT1_2)));
    #endif
}
