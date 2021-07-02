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

from .._definitions.types cimport DataType, LongIndexType, IndexType


# ============
# Declarations
# ============

cdef void py_generate_random_array(
        DataType* array,
        const LongIndexType array_size,
        const IndexType num_threads) except *
