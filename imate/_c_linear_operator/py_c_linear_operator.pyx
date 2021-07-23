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

import numpy
from libc.stdlib cimport exit

from .._definitions.types cimport DataType, ConstDataType, IndexType, \
        LongIndexType, FlagType
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
        self.long_index_type_name = NULL
        self.parameters = None

        # Check LongIndexType is a signed or unsigned type. If -1 overflows,
        # the type is unsigned.
        cdef LongIndexType long_index = -1
        if long_index < 1:
            unsigned_type = False
        else:
            unsigned_type = True

        # Set the long index type name
        if sizeof(LongIndexType) == 4:
            if unsigned_type:
                self.long_index_type_name = r'uint32'
            else:
                self.long_index_type_name = r'int32'
        elif sizeof(LongIndexType) == 8:
            if unsigned_type:
                self.long_index_type_name = r'uint64'
            else:
                self.long_index_type_name = r'int64'
        else:
            raise TypeError('"LongIndexType" has an unconventional byte size.')

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

    # ============
    # get num rows
    # ============

    cdef LongIndexType get_num_rows(self) except *:
        """
        :return Number of rows of matrix.
        :rtype: LongIdexType
        """

        if self.data_type_name == b'float32' and self.Aop_float != NULL:
            return self.Aop_float.get_num_rows()
        elif self.data_type_name == b'float64' and self.Aop_double != NULL:
            return self.Aop_double.get_num_rows()
        elif self.data_type_name == b'float128' and \
                self.Aop_long_double != NULL:
            return self.Aop_long_double.get_num_rows()
        else:
            raise ValueError('Linear operator is not set.')

    # ===============
    # get num columns
    # ===============

    cdef LongIndexType get_num_columns(self) except *:
        """
        :return Number of rows of matrix.
        :rtype: LongIdexType
        """

        if self.data_type_name == b'float32' and self.Aop_float != NULL:
            return self.Aop_float.get_num_columns()
        elif self.data_type_name == b'float64' and self.Aop_double != NULL:
            return self.Aop_double.get_num_columns()
        elif self.data_type_name == b'float128' and \
                self.Aop_long_double != NULL:
            return self.Aop_long_double.get_num_columns()
        else:
            raise ValueError('Linear operator is not set.')

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

    def get_data_type_name(self):
        """
        """

        if self.data_type_name == NULL:
            raise RuntimeError('Linear operator data type is not set.')

        return self.data_type_name

    # =========================
    # get linear operator float
    # =========================

    cdef cLinearOperator[float]* get_linear_operator_float(self) except *:
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

    cdef cLinearOperator[double]* get_linear_operator_double(self) except *:
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

    cdef cLinearOperator[long double]* get_linear_operator_long_double(
            self) except*:
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

    # ==============
    # set parameters
    # ==============

    def set_parameters(self, parameters_):
        """
        This function is only used for the test unit of this class. For the
        actual computations, the parameters are set though ``cLinearOperator``
        object directly, but not by this function.
        """

        if numpy.isscalar(parameters_):
            self.parameters = numpy.array([parameters_], dtype=float)
        else:
            self.parameters = parameters_

    # ===
    # dot
    # ===

    cpdef void dot(self, vector, product) except *:
        """
        """

        if vector.dtype != product.dtype:
            raise TypeError('The input vector and product should have ')

        # Declare memory views for input vector
        cdef float[:] mv_vector_float
        cdef double[:] mv_vector_double
        cdef long double[:] mv_vector_long_double

        # Declare memory views for output product
        cdef float[:] mv_product_float
        cdef double[:] mv_product_double
        cdef long double[:] mv_product_long_double

        # Declare memoryviews for parameters
        cdef float[:] mv_parameters_float
        cdef double[:] mv_parameters_double
        cdef long double[:] mv_parameters_long_double

        # Declare c pointers for input vector
        cdef float* c_vector_float
        cdef double* c_vector_double
        cdef long double* c_vector_long_double

        # Declare c pointers for output product
        cdef float* c_product_float
        cdef double* c_product_double
        cdef long double* c_product_long_double

        # Declare c pointers for parameters
        cdef float* c_parameters_float
        cdef double* c_parameters_double
        cdef long double* c_parameters_long_double

        # Dispatch to single, double or quadro precision
        if vector.dtype == 'float32':

            # input vector
            mv_vector_float = vector
            c_vector_float = &mv_vector_float[0]

            # output product
            mv_product_float = product
            c_product_float = &mv_product_float[0]

            # Set parameters
            if self.parameters is not None:
                mv_parameters_float = self.parameters.astype('float32')
                c_parameters_float = &mv_parameters_float[0]
                self.Aop_float.set_parameters(c_parameters_float)

            # Call c object
            self.Aop_float.dot(c_vector_float, c_product_float)

        elif vector.dtype == 'float64':

            # input vector
            mv_vector_double = vector
            c_vector_double = &mv_vector_double[0]

            # output product
            mv_product_double = product
            c_product_double = &mv_product_double[0]

            # Set parameters
            if self.parameters is not None:
                mv_parameters_double = self.parameters.astype('float64')
                c_parameters_double = &mv_parameters_double[0]
                self.Aop_double.set_parameters(c_parameters_double)

            # Call c object
            self.Aop_double.dot(c_vector_double, c_product_double)

        elif vector.dtype == 'float128':

            # input vector
            mv_vector_long_double = vector
            c_vector_long_double = &mv_vector_long_double[0]

            # output product
            mv_product_long_double = product
            c_product_long_double = &mv_product_long_double[0]

            # Set parameters
            if self.parameters is not None:
                mv_parameters_long_double = self.parameters.astype('float128')
                c_parameters_long_double = &mv_parameters_long_double[0]
                self.Aop_long_double.set_parameters(c_parameters_long_double)

            # Call c object
            self.Aop_long_double.dot(c_vector_long_double,
                                     c_product_long_double)

        else:
            raise TypeError('Vector type should be either "float32", ' +
                            '"float64", or "float128".')

    # =============
    # transpose dot
    # =============

    cpdef void transpose_dot(self, vector, product) except *:
        """
        """

        if vector.dtype != product.dtype:
            raise TypeError('The input vector and product should have ')

        # Declare memory views for input vector
        cdef float[:] mv_vector_float
        cdef double[:] mv_vector_double
        cdef long double[:] mv_vector_long_double

        # Declare memory views for output product
        cdef float[:] mv_product_float
        cdef double[:] mv_product_double
        cdef long double[:] mv_product_long_double

        # Declare memoryviews for parameters
        cdef float[:] mv_parameters_float
        cdef double[:] mv_parameters_double
        cdef long double[:] mv_parameters_long_double

        # Declare c pointers for input vector
        cdef float* c_vector_float
        cdef double* c_vector_double
        cdef long double* c_vector_long_double

        # Declare c pointers for output product
        cdef float* c_product_float
        cdef double* c_product_double
        cdef long double* c_product_long_double

        # Declare c pointers for parameters
        cdef float* c_parameters_float
        cdef double* c_parameters_double
        cdef long double* c_parameters_long_double

        # Dispatch to single, double or quadro precision
        if vector.dtype == 'float32':

            # input vector
            mv_vector_float = vector
            c_vector_float = &mv_vector_float[0]

            # output product
            mv_product_float = product
            c_product_float = &mv_product_float[0]

            # Set parameters
            if self.parameters is not None:
                mv_parameters_float = self.parameters.astype('float32')
                c_parameters_float = &mv_parameters_float[0]
                self.Aop_float.set_parameters(c_parameters_float)

            # Call c object
            self.Aop_float.transpose_dot(c_vector_float, c_product_float)

        elif vector.dtype == 'float64':

            # input vector
            mv_vector_double = vector
            c_vector_double = &mv_vector_double[0]

            # output product
            mv_product_double = product
            c_product_double = &mv_product_double[0]

            # Set parameters
            if self.parameters is not None:
                mv_parameters_double = self.parameters.astype('float64')
                c_parameters_double = &mv_parameters_double[0]
                self.Aop_double.set_parameters(c_parameters_double)

            # Call c object
            self.Aop_double.transpose_dot(c_vector_double, c_product_double)

        elif vector.dtype == 'float128':

            # input vector
            mv_vector_long_double = vector
            c_vector_long_double = &mv_vector_long_double[0]

            # output product
            mv_product_long_double = product
            c_product_long_double = &mv_product_long_double[0]

            # Set parameters
            if self.parameters is not None:
                mv_parameters_long_double = self.parameters.astype('float128')
                c_parameters_long_double = &mv_parameters_long_double[0]
                self.Aop_long_double.set_parameters(c_parameters_long_double)

            # Call c object
            self.Aop_long_double.transpose_dot(c_vector_long_double,
                                               c_product_long_double)

        else:
            raise TypeError('Vector type should be either "float32", ' +
                            '"float64", or "float128".')
