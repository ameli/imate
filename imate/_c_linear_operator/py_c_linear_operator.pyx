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

from libc.stdlib cimport exit
from libc.stdio cimport printf

from .._definitions.types cimport DataType, IndexType, LongIndexType, FlagType
from .c_linear_operator cimport cLinearOperator


# ===================
# pyc Linear Operator
# ===================

cdef class pycLinearOperator(object):
    """
    """

    # =========
    # __cinit__
    # =========

    def __cinit__(self):
        """
        Initializes attributes to zero.
        """

        # Initialize member data
        self.Aop_float = NULL
        self.Aop_double = NULL
        self.Aop_long_double = NULL
        self.data_type_name = NULL

    # ===========
    # __dealloc__
    # ===========

    def __dealloc__(self):
        """
        """

        if self.Aop_float != NULL:
            del self.Aop_float
            self.Aop_float = NULL

        if self.Aop_double != NULL:
            del self.Aop_double
            self.Aop_double = NULL

        if self.Aop_long_double != NULL:
            del self.Aop_long_double
            self.Aop_long_double = NULL

    # ==================
    # get num parameters
    # ==================

    def get_num_parameters(self):
        """
        :return: Number of parameters.
        :rtype: IndexType
        """

        if self.data_type_name == b'float32' and self.Aop_float != NULL:
            return self.Aop_float.get_num_parameters()
        elif self.data_type_name == b'float64' and self.Aop_double != NULL:
            return self.Aop_double.get_num_parameters()
        elif self.data_type_name == b'float128' and \
                self.Aop_long_double != NULL:
            return self.Aop_long_double.get_num_parameters()
        else:
            raise ValueError('Linear operator is not set.')

    # ==================
    # get data type name
    # ==================

    # cdef char* get_data_type_name(self):
    def get_data_type_name(self):
        """
        """

        if self.data_type_name == NULL:
            raise RuntimeError('Linear operator data type is not set.')

        return self.data_type_name

    # =========================
    # get linear operator float
    # =========================

    cdef cLinearOperator[float]* get_linear_operator_float(self):
        """
        """

        if self.Aop_float == NULL:
            raise RuntimeError('Linear operator (float type) is not set.')

        if self.data_type_name != b'float32':
            raise RuntimeError('Wrong accessors is called. The type of the ' +
                               'LinearOperator object is: %s'
                               % self.data_type_name)

        return self.Aop_float

    # ==========================
    # get linear operator double
    # ==========================

    cdef cLinearOperator[double]* get_linear_operator_double(self):
        """
        """

        if self.Aop_double == NULL:
            raise RuntimeError('Linear operator (double type) is not set.')

        if self.data_type_name != b'float64':
            raise RuntimeError('Wrong accessors is called. The type of the ' +
                               'LinearOperator object is: %s'
                               % self.data_type_name)

        return self.Aop_double

    # ===============================
    # get linear operator long double
    # ===============================

    cdef cLinearOperator[long double]* get_linear_operator_long_double(self):
        """
        """

        if self.Aop_long_double == NULL:
            raise RuntimeError('Linear operator (long double type) is not ' +
                               'set.')

        if self.data_type_name != b'float128':
            raise RuntimeError('Wrong accessors is called. The type of the ' +
                               'LinearOperator object is: %s'
                               % self.data_type_name)

        return self.Aop_long_double
