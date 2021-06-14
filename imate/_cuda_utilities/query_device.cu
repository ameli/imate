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
    // Query number of devices
    int num_devices;
    cudaError_t error = cudaGetDeviceCount(&num_devices);
    if (error != cudaSuccess)
    {
        return;
    }

    // Set number of devices
    device_properties.set_num_devices(num_devices);

    // Read properties of each device
    struct cudaDeviceProp properties;
    for (int device = 0; device < num_devices; ++device)
    {
        cudaGetDeviceProperties(&properties, device);

        // Machines with no GPUs may still report one emulation device
        if (properties.major == 9999)
        {
            // This is a gpu emulation not an actual device
            device_properties.num_multiprocessors[device] = 0;

            device_properties.num_threads_per_multiprocessor[device] = 0;
        }
        else
        {
            device_properties.num_multiprocessors[device] = \
                properties.multiProcessorCount;

            device_properties.num_threads_per_multiprocessor[device] = \
                properties.maxThreadsPerMultiProcessor;
        }
    }
}
