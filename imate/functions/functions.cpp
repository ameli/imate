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

#include "./functions.h"
#include <cmath>  // log, exp, pow, tanh, M_SQRT1_2, M_2_SQRTPI, NAN


// ===================
// Function destructor
// ===================

/// \brief Default virtual destructor.
///

Function::~Function()
{
}
