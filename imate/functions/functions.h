/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */


#ifndef FUNCTIONS_FUNCTIONS_H_
#define FUNCTIONS_FUNCTIONS_H_

// ========
// Function
// ========

/// \brief   Defines the function \f$ f: \lambda \mapsto \lambda \f$.
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
///
/// \note    This class is a base class for other classes and serves as an
///          interface. To create a new matrix function, derive a class from
///          \c Function class and implement the \c function method.

class Function
{
    public:
        virtual ~Function();
        virtual float function(const float lambda_) const = 0;
        virtual double function(const double lambda_) const = 0;
        virtual long double function(const long double lambda_) const = 0;
};

#endif  // FUNCTIONS_FUNCTIONS_H_
