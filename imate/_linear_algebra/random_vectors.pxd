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

from .._definitions.types cimport DataType, IndexType, LongIndexType


# ============
# Declarations
# ============

# generate random column vectors
cdef void generate_random_column_vectors(
        DataType* vectors,
        const LongIndexType vector_size,
        const IndexType num_vectors,
        const IndexType orthogonalize,
        const IndexType num_parallel_threads)
