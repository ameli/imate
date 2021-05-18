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

#include "./query_device.h"


// =================
// Device Properties
// =================

/// \brief Constructor
///

DeviceProperties::DeviceProperties():
    num_devices(0),
    num_multiprocessors(0),
    num_threads_per_multiprocessor(0)
{
}


// ============
// query device
// ============

/// \brief      Queries GPU device information, such as the number of devices,
///             number of multiprocessors, and the number of threads per each
///             multiprocessor.
///
/// \param[out] device_properties
///             A struct to be filled with the number of devices, threads and
///             multiprocessors.

void query_device(DeviceProperties& device_properties)
{
    cudaError_t error = cudaGetDeviceCount(&device_properties.num_devices);
    if (error != cudaSuccess)
    {
        return;
    }

    // Machines with no GPUs may still report one emulation device
    struct cudaDeviceProp properties;
    for (int device = 0; device < device_properties.num_devices; ++device)
    {
        cudaGetDeviceProperties(&properties, device);

        // exclude gpu emulation
        if (properties.major != 9999)
        {
            if (device == 0)
            {
                device_properties.num_multiprocessors = \
                    properties.multiProcessorCount;

                device_properties.num_threads_per_multiprocessor = \
                    properties.maxThreadsPerMultiProcessor;
            }
        }
    }
}
