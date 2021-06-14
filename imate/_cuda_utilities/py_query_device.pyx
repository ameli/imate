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

import numpy
from .query_device cimport query_device, DeviceProperties


# ===============
# py query device
# ===============

cdef py_query_device():
    """
    A python wrapper for ``query_device()`` function. This function queries
    the gpu device(s) and returns a dictionary.

    :return: A dictionary that contains the following fields:
        * ``num_devices``: number of gpu devices.
        * ``num_multiprocessors``: Number of multiprocessors per device.
        * ``num_threads_per_multiprocessor``: Number of threads per
          multiprocessor.
    :rtype: dict
    """

    # Create and fill a C class
    cdef DeviceProperties device_properties
    query_device(device_properties)

    # Declare arrays to hold data for each gpu device
    num_multiprocessors = numpy.empty(
            (device_properties.num_devices, ), dtype=int)
    num_threads_per_multiprocessor = numpy.empty(
            (device_properties.num_devices, ), dtype=int)

    for device in range(device_properties.num_devices):
        num_multiprocessors[device] = \
            device_properties.num_multiprocessors[device]
        num_threads_per_multiprocessor[device] = \
            device_properties.num_threads_per_multiprocessor[device]

    # Convert to python class
    device_properties_dict = {
        'num_devices': device_properties.num_devices,
        'num_multiprocessors': num_multiprocessors,
        'num_threads_per_multiprocessor': num_threads_per_multiprocessor
    }

    return device_properties_dict
