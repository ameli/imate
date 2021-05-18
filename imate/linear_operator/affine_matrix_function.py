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


# ======================
# Affine Matrix Function
# ======================

class AffineMatrixFunction(LinearOperator):
    """
    """

    # ====
    # init
    # ====

    def __init__(self, A_, B_=None):
        """
        """

        # Calling base class to initialize member data
        super(AffineMatrixFunction, self).__init__()

        # A reference to numpy or scipt sparse matrices
        self.A = A_
        self.B = B_
        self.num_parameters = 1
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
                from .._cu_linear_operator import pycuAffineMatrixFunction
                self.gpu_Aop = pycuAffineMatrixFunction(self.A, self.B)
                self.initialized_on_gpu = True

        else:

            if not self.initialized_on_cpu:

                # Create a lienar operator on CPU
                from .._c_linear_operator import pycAffineMatrixFunction
                self.cpu_Aop = pycAffineMatrixFunction(self.A, self.B)
                self.initialized_on_cpu = True
