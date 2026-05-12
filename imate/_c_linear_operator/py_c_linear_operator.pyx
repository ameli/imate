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

from .._definitions.types cimport LongIndexType
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
        self.Aop_fp32 = NULL
        self.Aop_fp64 = NULL
        self.Aop_fp128 = NULL
        self.data_type_name = ""
        self.long_index_type_name = ""
        self.parameters = None

        # Check LongIndexType is a signed or unsigned type. If -1 overflows,
        # the type is unsigned.
        cdef LongIndexType long_index = -1
        if long_index < 0:
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
            raise TypeError('"LongIndexType" has unconventional byte size.')

    # ===========
    # __dealloc__
    # ===========

    def __dealloc__(self):
        """
        """

        if self.Aop_fp32 != NULL:
            del self.Aop_fp32
            self.Aop_fp32 = NULL

        if self.Aop_fp64 != NULL:
            del self.Aop_fp64
            self.Aop_fp64 = NULL

        if self.Aop_fp128 != NULL:
            del self.Aop_fp128
            self.Aop_fp128 = NULL

    # ============
    # get num rows
    # ============

    cdef LongIndexType get_num_rows(self) except *:
        """
        :return Number of rows of matrix.
        :rtype: LongIdexType
        """

        if (self.data_type_name == b'float32') and (self.Aop_fp32 != NULL):
            return self.Aop_fp32.get_num_rows()
        elif (self.data_type_name == b'float64') and (self.Aop_fp64 != NULL):
            return self.Aop_fp64.get_num_rows()
        elif (self.data_type_name == b'float128') and \
                (self.Aop_fp128 != NULL):
            return self.Aop_fp128.get_num_rows()
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

        if (self.data_type_name == b'float32') and (self.Aop_fp32 != NULL):
            return self.Aop_fp32.get_num_columns()
        elif (self.data_type_name == b'float64') and (self.Aop_fp64 != NULL):
            return self.Aop_fp64.get_num_columns()
        elif (self.data_type_name == b'float128') and \
                (self.Aop_fp128 != NULL):
            return self.Aop_fp128.get_num_columns()
        else:
            raise ValueError('Linear operator is not set.')

    # ==================
    # get num parameters
    # ==================

    def get_num_parameters(self):
        """
        :return: Number of parameters.
        :rtype: int
        """

        if (self.data_type_name == b'float32') and (self.Aop_fp32 != NULL):
            return self.Aop_fp32.get_num_parameters()
        elif (self.data_type_name == b'float64') and (self.Aop_fp64 != NULL):
            return self.Aop_fp64.get_num_parameters()
        elif (self.data_type_name == b'float128') and \
                (self.Aop_fp128 != NULL):
            return self.Aop_fp128.get_num_parameters()
        else:
            raise ValueError('Linear operator is not set.')

    # ==================
    # get data type name
    # ==================

    def get_data_type_name(self):
        """
        """

        # if self.data_type_name == NULL:
        if self.data_type_name == "":
            raise RuntimeError('Linear operator data type is not set.')

        return self.data_type_name

    # ========================
    # get linear operator fp32
    # ========================

    cdef cLinearOperator[float]* get_linear_operator_fp32(self) except *:
        """
        """

        if self.Aop_fp32 == NULL:
            raise RuntimeError('Linear operator (float type) is not set.')

        if self.data_type_name != b'float32':
            raise RuntimeError('Wrong accessors is called. The type of the ' +
                               'LinearOperator object is: %s'
                               % self.data_type_name)

        return self.Aop_fp32

    # ========================
    # get linear operator fp64
    # ========================

    cdef cLinearOperator[double]* get_linear_operator_fp64(self) except *:
        """
        """

        if self.Aop_fp64 == NULL:
            raise RuntimeError('Linear operator (double type) is not set.')

        if self.data_type_name != b'float64':
            raise RuntimeError('Wrong accessors is called. The type of the ' +
                               'LinearOperator object is: %s'
                               % self.data_type_name)

        return self.Aop_fp64

    # =========================
    # get linear operator fp128
    # =========================

    cdef cLinearOperator[long double]* get_linear_operator_fp128(
            self) except*:
        """
        """

        if self.Aop_fp128 == NULL:
            raise RuntimeError('Linear operator (long double type) is not ' +
                               'set.')

        if self.data_type_name != b'float128':
            raise RuntimeError('Wrong accessors is called. The type of the ' +
                               'LinearOperator object is: %s'
                               % self.data_type_name)

        return self.Aop_fp128

    # ============
    # set symmetry
    # ============

    cpdef void set_symmetry(self, symmetric) except *:
        """
        Sets the matrix A (or matrices A and B in case of affine operator) to
        be symmetric or non-symmetric.
        """

        if not isinstance(symmetric, bool):
            raise ValueError('"symmetric" should be boolean.')

        symmetric = int(symmetric)

        if (self.data_type_name == b'float32') and (self.Aop_fp32 != NULL):
            self.Aop_fp32.set_symmetry(symmetric)
            
        elif (self.data_type_name == b'float64') and (self.Aop_fp64 != NULL):
            self.Aop_fp64.set_symmetry(symmetric)
            
        elif (self.data_type_name == b'float128') and \
                (self.Aop_fp128 != NULL):
            self.Aop_fp128.set_symmetry(symmetric)

        else:
            raise ValueError('Linear operator is not set.')


    # ==============
    # set parameters
    # ==============

    cpdef void set_parameters(self, parameters) except *:
        """
        This function is only used for the test unit of this class. For the
        actual computations, the parameters are set though ``cLinearOperator``
        object directly, but not by this function.
        """

        if numpy.isscalar(parameters):
            self.parameters = numpy.array([parameters], dtype=float)
        elif isinstance(parameters, (list, tuple)):
            self.parameters = numpy.array(parameters, dtype=float)
        else:
            self.parameters = parameters

        # Declare memory views for parameters
        cdef float[:] mv_parameters_fp32
        cdef double[:] mv_parameters_fp64
        cdef long double[:] mv_parameters_fp128

        # Declare c pointers for parameters
        cdef float* c_parameters_fp32
        cdef double* c_parameters_fp64
        cdef long double* c_parameters_fp128

        if (self.data_type_name == b'float32') and (self.Aop_fp32 != NULL):
            mv_parameters_fp32 = self.parameters.astype('float32')
            c_parameters_fp32 = &mv_parameters_fp32[0]
            self.Aop_fp32.set_parameters(c_parameters_fp32)
            
        elif (self.data_type_name == b'float64') and (self.Aop_fp64 != NULL):
            mv_parameters_fp64 = self.parameters.astype('float64')
            c_parameters_fp64 = &mv_parameters_fp64[0]
            self.Aop_fp64.set_parameters(c_parameters_fp64)
            
        elif (self.data_type_name == b'float128') and \
                (self.Aop_fp128 != NULL):
            mv_parameters_fp128 = self.parameters.astype('float128')
            c_parameters_fp128 = &mv_parameters_fp128[0]
            self.Aop_fp128.set_parameters(c_parameters_fp128)

        else:
            raise ValueError('Linear operator is not set.')

    # ===
    # dot
    # ===

    cpdef void dot(self, vector, product) except *:
        """
        """

        # Make sure input vector and matrix have the same data type
        if vector.dtype != self.data_type_name:
            vector_typed = vector.astype(self.data_type_name)
        else:
            vector_typed = vector

        if vector_typed.dtype != product.dtype:
            raise TypeError('The input and output vectors should have the '
                            'same data type.')

        # Declare memory views for input vector
        cdef float[:] mv_vector_fp32
        cdef double[:] mv_vector_fp64
        cdef long double[:] mv_vector_fp128

        # Declare memory views for output product
        cdef float[:] mv_product_fp32
        cdef double[:] mv_product_fp64
        cdef long double[:] mv_product_fp128

        # Declare c pointers for input vector
        cdef float* c_vector_fp32
        cdef double* c_vector_fp64
        cdef long double* c_vector_fp128

        # Declare c pointers for output product
        cdef float* c_product_fp32
        cdef double* c_product_fp64
        cdef long double* c_product_fp128

        # Dispatch to single, double or quadro precision
        if vector_typed.dtype == 'float32':

            # input vector
            mv_vector_fp32 = vector_typed
            c_vector_fp32 = &mv_vector_fp32[0]

            # output product
            mv_product_fp32 = product
            c_product_fp32 = &mv_product_fp32[0]

            # Call c object
            self.Aop_fp32.dot(c_vector_fp32, c_product_fp32)

        elif vector_typed.dtype == 'float64':

            # input vector
            mv_vector_fp64 = vector_typed
            c_vector_fp64 = &mv_vector_fp64[0]

            # output product
            mv_product_fp64 = product
            c_product_fp64 = &mv_product_fp64[0]

            # Call c object
            self.Aop_fp64.dot(c_vector_fp64, c_product_fp64)

        elif vector_typed.dtype == 'float128':

            # input vector
            mv_vector_fp128 = vector_typed
            c_vector_fp128 = &mv_vector_fp128[0]

            # output product
            mv_product_fp128 = product
            c_product_fp128 = &mv_product_fp128[0]

            # Call c object
            self.Aop_fp128.dot(c_vector_fp128, c_product_fp128)

        else:
            raise TypeError('Vector type should be either "float32", ' +
                            '"float64", or "float128".')

    # =============
    # transpose dot
    # =============

    cpdef void transpose_dot(self, vector, product) except *:
        """
        """

        # Make sure input vector and matrix have the same data type
        if vector.dtype != self.data_type_name:
            vector_typed = vector.astype(self.data_type_name)
        else:
            vector_typed = vector

        if vector_typed.dtype != product.dtype:
            raise TypeError('The input and output vectors should have the '
                            'same data type.')

        # Declare memory views for input vector
        cdef float[:] mv_vector_fp32
        cdef double[:] mv_vector_fp64
        cdef long double[:] mv_vector_fp128

        # Declare memory views for output product
        cdef float[:] mv_product_fp32
        cdef double[:] mv_product_fp64
        cdef long double[:] mv_product_fp128

        # Declare c pointers for input vector
        cdef float* c_vector_fp32
        cdef double* c_vector_fp64
        cdef long double* c_vector_fp128

        # Declare c pointers for output product
        cdef float* c_product_fp32
        cdef double* c_product_fp64
        cdef long double* c_product_fp128

        # Dispatch to single, double or quadro precision
        if vector_typed.dtype == 'float32':

            # input vector
            mv_vector_fp32 = vector_typed
            c_vector_fp32 = &mv_vector_fp32[0]

            # output product
            mv_product_fp32 = product
            c_product_fp32 = &mv_product_fp32[0]

            # Call c object
            self.Aop_fp32.transpose_dot(c_vector_fp32, c_product_fp32)

        elif vector_typed.dtype == 'float64':

            # input vector
            mv_vector_fp64 = vector_typed
            c_vector_fp64 = &mv_vector_fp64[0]

            # output product
            mv_product_fp64 = product
            c_product_fp64 = &mv_product_fp64[0]

            # Call c object
            self.Aop_fp64.transpose_dot(c_vector_fp64, c_product_fp64)

        elif vector_typed.dtype == 'float128':

            # input vector
            mv_vector_fp128 = vector_typed
            c_vector_fp128 = &mv_vector_fp128[0]

            # output product
            mv_product_fp128 = product
            c_product_fp128 = &mv_product_fp128[0]

            # Call c object
            self.Aop_fp128.transpose_dot(c_vector_fp128, c_product_fp128)

        else:
            raise TypeError('Vector type should be either "float32", ' +
                            '"float64", or "float128".')
