/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */


#ifndef _RANDOM_GENERATOR_HIGHRES_TIME_STAMP_H_
#define _RANDOM_GENERATOR_HIGHRES_TIME_STAMP_H_


// ======
// Header
// ======

#include <stdint.h>  // uint64_t


// ============
// Declarations
// ============

// Get HighRes Time Stamp
uint64_t get_highres_time_stamp(void);


#endif  // _RANDOM_GENERATOR_HIGHRES_TIME_STAMP_H_
