# SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the license found in the LICENSE.txt file in the root directory
# of this source tree.


# =======
# Imports
# =======

from .query_device cimport query_device, DeviceProperties


# ===============
# py query device
# ===============

cdef py_query_device():
    """
    """

    # Create and fill a C class
    cdef DeviceProperties device_properties
    query_device(device_properties)

    # Convert to python class
    device_properties_dict = {
        'num_devices': device_properties.num_devices,
        'num_multiprocessors': device_properties.num_multiprocessors,
        'num_threads_per_multiprocessor':
            device_properties.num_threads_per_multiprocessor
    }

    return device_properties_dict
