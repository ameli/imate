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

    def initialize(self, gpu, num_gpu_devices):
        """
        """

        if gpu:

            if not self.initialized_on_gpu:

                # Create a linear operator on GPU
                from .._cu_linear_operator import pycuAffineMatrixFunction
                self.gpu_Aop = pycuAffineMatrixFunction(self.A, self.B,
                                                        num_gpu_devices)
                self.initialized_on_gpu = True

            elif num_gpu_devices != self.num_gpu_devices:

                # Get the actual (all) number of gpu devices
                device_properties_dict = self.gpu_Aop.get_device_properties()
                num_all_gpu_devices = device_properties_dict['num_devices']

                # If number of gpu devices is zero, it means all gpu devices
                # were used. Exclude cases when one of num_gpu_devices and
                # self.num_gpu_devices is zero and the other is equal to all
                # gpu devices to avoid unnecessary reallocation.
                reallocate = True
                if (num_gpu_devices == 0) and \
                   (self.num_gpu_devices == num_all_gpu_devices):
                    reallocate = False
                elif (num_gpu_devices == num_all_gpu_devices) and \
                     (self.num_gpu_devices == 0):
                    reallocate = False

                if reallocate:

                    # Create a linear operator on GPU
                    from .._cu_linear_operator import pycuAffineMatrixFunction

                    # If allocated before, deallocate. This occurs then the
                    # number of gpu devices changes and gpu_Aop needs to be
                    # reallocated.
                    if self.gpu_Aop is not None:
                        self.gpu_Aop.__dealloc__()

                    self.gpu_Aop = pycuAffineMatrixFunction(self.A, self.B,
                                                            num_gpu_devices)
                    self.initialized_on_gpu = True

        else:

            if not self.initialized_on_cpu:

                # Create a linear operator on CPU
                from .._c_linear_operator import pycAffineMatrixFunction
                self.cpu_Aop = pycAffineMatrixFunction(self.A, self.B)
                self.initialized_on_cpu = True

        self.gpu = gpu
        self.num_gpu_devices = num_gpu_devices

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
