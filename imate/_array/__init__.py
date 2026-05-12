# SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the license found in the LICENSE.txt file in the root
# directory of this source tree.


from .array_properties import get_device, get_shape, get_size, get_ndim, \
    get_data_type_name, is_row_major

__all__ = ['get_device', 'get_shape', 'get_size', 'get_ndim',
           'get_data_type_name', 'is_row_major']
