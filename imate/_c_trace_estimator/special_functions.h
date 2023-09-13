/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */


#ifndef _C_TRACE_ESTIMATOR_SPECIAL_FUNCTIONS_H_
#define _C_TRACE_ESTIMATOR_SPECIAL_FUNCTIONS_H_

// ======
// Headers
// ======

#include "../_definitions/types.h"  // DataType, IndexType, FlagType

// ============
// Declarations
// ============

// erf inv
double erf_inv(const double x);

#endif  // _C_TRACE_ESTIMATOR_SPECIAL_FUNCTIONS_H_
