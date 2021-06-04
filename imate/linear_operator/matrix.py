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

from .linear_operator import LinearOperator


# ======
# Matrix
# ======

class Matrix(LinearOperator):
    """
    """

    # ====
    # init
    # ====

    def __init__(self, A_):
        """
        """

        # Calling base class to initialize member data
        super(Matrix, self).__init__()

        # A reference to numpy or scipt sparse matrix
        self.A = A_
        self.num_parameters = 0
        self.set_data_type_name(self.A)

    # ==========
    # initialize
    # ==========

    def initialize(self):
        """
        """

        if self.gpu:

            if not self.initialized_on_gpu:

                # Create a lienar operator on GPU
                from .._cu_linear_operator import pycuMatrix
                self.gpu_Aop = pycuMatrix(self.A)
                self.initialized_on_gpu = True

        else:

            if not self.initialized_on_cpu:

                # Create a lienar operator on CPU
                from .._c_linear_operator import pycMatrix
                self.cpu_Aop = pycMatrix(self.A)
                self.initialized_on_cpu = True

    # ============
    # get num rows
    # ============

    def get_num_rows(self):
        """
        """

        return self.A.shape[0]

    # ===============
    # get num columns
    # ===============

    def get_num_columns(self):
        """
        """

        return self.A.shape[1]
