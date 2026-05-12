# SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the license found in the LICENSE.txt file in the root directory
# of this source tree.


# ============
# Static types (non-templates)
# ============

# The LongIndexType is defined in definitions.h and is used for the C++ source
# codes. This type can be 4 byte int or 8 byte long int, depending on macros
# that is defined at compile time. To expose the LongIndexType to pyx files as
# well, we extern the definitions.h. But this also requires to re-define this
# type with ctypedef and with yet another type (here, int), despite it might
# be defined as long int in definitions.h. Note that the int type here is
# arbitrary and will be ignored by the cython compiler. Read more about this at
# https://cython.readthedocs.io/en/latest/src/userguide/external_C_code.html
# in the typedef section of that page.

cdef extern from "./cu_types.h":
    ctypedef struct __nv_fp8_e5m2
    ctypedef struct __nv_fp8_e4m3
    ctypedef struct __half
    ctypedef struct __nv_bfloat16
