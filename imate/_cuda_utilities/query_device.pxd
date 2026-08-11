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

from .device_properties cimport DeviceProperties


# =======
# Externs
# =======

cdef extern from "query_device.h":

    cdef int query_device(DeviceProperties& device_properties) noexcept nogil
