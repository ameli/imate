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

from .._definitions.types cimport IndexType, LongIndexType, FlagType


# =======
# Externs
# =======

# cdef extern from "cu_orthogonalization.h":
#
#     cdef cppclass cuOrthogonalization[DataType]:
#
#         # Gram-Schmidt Process
#         @staticmethod
#         void gram_schmidt_process(
#                 const DataType* V,
#                 const LongIndexType vector_size,
#                 const IndexType num_vectors,
#                 const IndexType last_vector,
#                 const FlagType num_ortho,
#                 DataType* r) nogil
#
#         # Orthogonalize Vectors
#         @staticmethod
#         void orthogonalize_vectors(
#                 DataType* vectors,
#                 const LongIndexType vector_size,
#                 const IndexType num_vectors) nogil
