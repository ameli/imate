/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */


#ifndef _DEFINITIONS_DEFINITIONS_H_
#define _DEFINITIONS_DEFINITIONS_H_


// ===========
// Definitions
// ===========

// If set to 0, the LongIndexType is declared as 32-bit integer. Whereas if set
// to 1, the LongIndexType is declared as 64-bit integer.
#ifndef LONG_INT
    #define LONG_INT 0
#endif

// If set to 0, the LongIndexType is declared as signed integer, whereas if set
// to 1, the LongIndexType is declared as unsigned integer.
#ifndef UNSIGNED_LONG_INT
    #define UNSIGNED_LONG_INT 0
#endif


#endif  // _DEFINITIONS_DEFINITIONS_H_
