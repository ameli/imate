# SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the license found in the LICENSE.txt file in the root
# directory of this source tree.


# ===============
# Linear Operator
# ===============

class LinearOperator(object):
    """
    """

    # ====
    # init
    # ====

    def __init__(self):
        """
        """

        self.Aop = None
        self.cpu_Aop = None
        self.gpu_Aop = None
        self.gpu = False
        self.num_gpu_devices = 0
        self.initialized_on_cpu = False
        self.initialized_on_gpu = False
        self.data_type_name = None
        self.num_parameters = None

    # ==========
    # Initialize
    # ==========

    def initialize(self):
        """
        """

        raise NotImplementedError('This method should be called by a derived' +
                                  ' class.')

    # ============
    # get num rows
    # ============

    def get_num_rows(self):
        """
        """

        raise NotImplementedError('This method should be called by a derived' +
                                  ' class.')

    # ===============
    # get num columns
    # ===============

    def get_num_columns(self):
        """
        """

        raise NotImplementedError('This method should be called by a derived' +
                                  ' class.')

    # =========
    # is sparse
    # =========

    def is_sparse(self):
        """
        """

        raise NotImplementedError('This method should be called by a derived' +
                                  ' class.')

    # =======
    # get nnz
    # =======

    def get_nnz(self):
        """
        """

        raise NotImplementedError('This method should be called by a derived' +
                                  ' class.')

    # ===========
    # get density
    # ===========

    def get_density(self):
        """
        """

        raise NotImplementedError('This method should be called by a derived' +
                                  ' class.')

    # ===================
    # get linear operator
    # ===================

    def get_linear_operator(self, gpu=False, num_gpu_devices=0):
        """
        """

        # Initialize matrix either on cpu or gpu (implemented in sub-classes)
        self.initialize(gpu, num_gpu_devices)

        if gpu:
            if self.gpu_Aop is None or not self.initialized_on_gpu:
                raise RuntimeError('Matrix is not initialized on gpu.')
            self.Aop = self.gpu_Aop
        else:
            if self.cpu_Aop is None or not self.initialized_on_cpu:
                raise RuntimeError('Matrix is not initialized on cpu.')
            self.Aop = self.cpu_Aop

        return self.Aop

    # ==================
    # set data type name
    # ==================

    def set_data_type_name(self, A):
        """
        """

        # data type name
        if A.dtype == b'float32':
            self.data_type_name = b'float32'

        elif A.dtype == b'float64':
            self.data_type_name = b'float64'

        elif A.dtype == b'float128':
            self.data_type_name = b'float128'

        else:
            raise TypeError('Data type should be "float32", "float64", or ' +
                            '"float128".')

    # ==================
    # get data type name
    # ==================

    def get_data_type_name(self):
        """
        """

        if self.data_type_name is None:
            raise RuntimeError('"data_type_name" is None.')

        return self.data_type_name

    # ==================
    # get num parameters
    # ==================

    def get_num_parameters(self):
        """
        """

        if self.num_parameters is None:
            raise RuntimeError('"num_parameters" is None.')

        return self.num_parameters
