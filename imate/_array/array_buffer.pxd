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

# get array buffer
cdef const void* get_array_buffer(array, Py_buffer* py_buffer)

# release buffer
cdef void release_buffer(Py_buffer* py_buffer)
