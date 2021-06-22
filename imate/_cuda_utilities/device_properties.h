/*
 *  SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
 *  SPDX-License-Identifier: BSD-3-Clause
 *  SPDX-FileType: SOURCE
 *
 *  This program is free software: you can redistribute it and/or modify it
 *  under the terms of the license found in the LICENSE.txt file in the root
 *  directory of this source tree.
 */

#ifndef _CUDA_UTILITIES_DEVICE_PROPERTIES_H_
#define _CUDA_UTILITIES_DEVICE_PROPERTIES_H_


// =================
// Device Properties
// =================

/// \class DeviceProperties
///
/// \brief Properties of GPU devices.

struct DeviceProperties
{
    // Methods
    DeviceProperties();
    ~DeviceProperties();
    void deallocate_members();
    void set_num_devices(int num_devices_);

    // Data
    int num_devices;
    int* num_multiprocessors;
    int* num_threads_per_multiprocessor;
};

#endif  // _CUDA_UTILITIES_DEVICE_PROPERTIES_H_
