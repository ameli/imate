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

#include "./device_properties.h"
#include <cstdlib>  // NULL


// ===========
// Constructor
// ===========

/// \brief Constructor
///

DeviceProperties::DeviceProperties():
    num_devices(0),
    num_multiprocessors(NULL),
    num_threads_per_multiprocessor(NULL)
{
}


// ==========
// Destructor
// ==========

/// \brief Destructor
///

DeviceProperties::~DeviceProperties()
{
    this->deallocate_members();
}


// ==================
// deallocate members
// ==================

/// \brief Deallocates the member data.
///

void DeviceProperties::deallocate_members()
{
    if (this->num_multiprocessors != NULL)
    {
        delete[] this->num_multiprocessors;
        this->num_multiprocessors = NULL;
    }

    if (this->num_threads_per_multiprocessor != NULL)
    {
        delete[] this->num_threads_per_multiprocessor;
        this->num_threads_per_multiprocessor = NULL;
    }
}


// ===============
// set num devices
// ===============

/// \brief      Sets the number of devices and allocates memory for member data
///             with the size of devices.
///
/// \param[in] num_devices_
///            Number of gpu devices.

void DeviceProperties::set_num_devices(int num_devices_)
{
    this->num_devices = num_devices_;

    // Deallocate members in case they were allocated before
    this->deallocate_members();

    // Allocate members
    this->num_multiprocessors = new int[this->num_devices];
    this->num_threads_per_multiprocessor = new int[this->num_devices];
}
