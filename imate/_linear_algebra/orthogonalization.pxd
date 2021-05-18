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

from .._definitions.types cimport DataType, IndexType, LongIndexType, FlagType


# ============
# Declarations
# ============

# Gram-Schmidt Process
cdef void gram_schmidt_process(
        const DataType* V,
        const LongIndexType vector_size,
        const IndexType num_vectors,
        const FlagType ortho_depth,
        DataType* r) nogil

# Orthogonalize Vectors
cdef void orthogonalize_vectors(
        DataType* vectors,
        const LongIndexType vector_size,
        const IndexType num_vectors) nogil
