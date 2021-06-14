/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */

#ifndef _CUDA_UTILITIES_QUERY_DEVICE_H_
#define _CUDA_UTILITIES_QUERY_DEVICE_H_


// =======
// Headers
// =======

#include "./device_properties.h"


// ============
// Declarations
// ============

void query_device(DeviceProperties& device_properties);

#endif  // _CUDA_UTILITIES_QUERY_DEVICE_H_
