# SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the license found in the LICENSE.txt file in the root
# directory of this source tree.


# ============
# Declarations
# ============

cdef double matern_kernel(
        const double x,
        const double correlation_scale,
        const double nu) nogil

cdef double euclidean_distance(
        const double[:] point1,
        const double[:] point2,
        const int dimension) nogil
