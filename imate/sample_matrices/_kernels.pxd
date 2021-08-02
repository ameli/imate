# SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the license found in the LICENSE.txt file in the root
# directory of this source tree.


# =======
# Imports
# =======

from .._definitions.types cimport kernel_type


# ============
# Declarations
# ============

cdef kernel_type get_kernel(const char* kernel_type)

cdef double _matern_kernel(
        const double x,
        const double param) nogil

cdef double _exponential_kernel(
        const double x,
        const double param) nogil

cdef double _square_exponential_kernel(
        const double x,
        const double param) nogil

cdef double _rational_quadratic_kernel(
        const double x,
        const double param) nogil

cdef double euclidean_distance(
        const double[:] point1,
        const double[:] point2,
        const double distance_scale,
        const int dimension) nogil
