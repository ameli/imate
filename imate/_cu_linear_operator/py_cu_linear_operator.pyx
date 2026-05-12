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
from .cu_linear_operator cimport cuLinearOperator
from .._cuda_utilities cimport py_query_device


# ====================
# pycu Linear Operator
# ====================

cdef class pycuLinearOperator(object):
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
        self.Aop_fp8_e5m2 = NULL
        self.Aop_fp8_e4m3 = NULL
        self.Aop_fp16 = NULL
        self.Aop_bf16 = NULL
        self.Aop_fp32 = NULL
        self.Aop_fp64 = NULL
        self.data_type_name = ""
        self.long_index_type_name = ""
        self.parameters = None
        self.num_gpu_devices = 0
        self.device_properties_dict = py_query_device()

        # Check if GPU device is found. If not, this will raise exception
        # sooner than the C++ code aborting the whole process.
        num_all_gpu_devices = self.device_properties_dict['num_devices']
        if num_all_gpu_devices == 0:
            raise RuntimeError('No cuda-capable gpu device was found.')

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

        if self.Aop_fp8_e5m2 != NULL:
            del self.Aop_fp8_e5m2
            self.Aop_fp8_e5m2 = NULL

        if self.Aop_fp8_e4m3 != NULL:
            del self.Aop_fp8_e4m3
            self.Aop_fp8_e4m3 = NULL

        if self.Aop_fp16 != NULL:
            del self.Aop_fp16
            self.Aop_fp16 = NULL

        if self.Aop_bf16 != NULL:
            del self.Aop_bf16
            self.Aop_bf16 = NULL

        if self.Aop_fp32 != NULL:
            del self.Aop_fp32
            self.Aop_fp32 = NULL

        if self.Aop_fp64 != NULL:
            del self.Aop_fp64
            self.Aop_fp64 = NULL

    # ============
    # get num rows
    # ============

    cdef LongIndexType get_num_rows(self) except *:
        """
        :return Number of rows of matrix.
        :rtype: LongIdexType
        """

        if (self.data_type_name == b'float8_e5m2') and \
                (self.Aop_fp8_e5m2 != NULL):
            return self.Aop_fp8_e5m2.get_num_rows()
        elif (self.data_type_name == b'float8_e4m3') and \
                (self.Aop_fp8_e4m3 != NULL):
            return self.Aop_fp8_e4m3.get_num_rows()
        elif (self.data_type_name == b'float16') and (self.Aop_fp16 != NULL):
            return self.Aop_fp16.get_num_rows()
        elif (self.data_type_name == b'bfloat16') and (self.Aop_bf16 != NULL):
            return self.Aop_bf16.get_num_rows()
        elif (self.data_type_name == b'float32') and (self.Aop_fp32 != NULL):
            return self.Aop_fp32.get_num_rows()
        elif (self.data_type_name == b'float64') and (self.Aop_fp64 != NULL):
            return self.Aop_fp64.get_num_rows()
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

        if (self.data_type_name == b'float8_e5m2') and \
                (self.Aop_fp8_e5m2 != NULL):
            return self.Aop_fp8_e5m2.get_num_columns()
        elif (self.data_type_name == b'float8_e4m3') and \
                (self.Aop_fp8_e4m3 != NULL):
            return self.Aop_fp8_e4m3.get_num_columns()
        elif (self.data_type_name == b'float16') and (self.Aop_fp16 != NULL):
            return self.Aop_fp16.get_num_columns()
        elif (self.data_type_name == b'bfloat16') and (self.Aop_bf16 != NULL):
            return self.Aop_bf16.get_num_columns()
        elif (self.data_type_name == b'float32') and (self.Aop_fp32 != NULL):
            return self.Aop_fp32.get_num_columns()
        elif (self.data_type_name == b'float64') and (self.Aop_fp64 != NULL):
            return self.Aop_fp64.get_num_columns()
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

        if (self.data_type_name == b'float8_e5m2') and \
                (self.Aop_fp8_e5m2 != NULL):
            return self.Aop_fp8_e5m2.get_num_parameters()
        elif (self.data_type_name == b'float8_e4m3') and \
                (self.Aop_fp8_e4m3 != NULL):
            return self.Aop_fp8_e4m3.get_num_parameters()
        elif (self.data_type_name == b'float16') and (self.Aop_fp16 != NULL):
            return self.Aop_fp16.get_num_parameters()
        elif (self.data_type_name == b'bfloat16') and (self.Aop_bf16 != NULL):
            return self.Aop_bf16.get_num_parameters()
        elif (self.data_type_name == b'float32') and (self.Aop_fp32 != NULL):
            return self.Aop_fp32.get_num_parameters()
        elif (self.data_type_name == b'float64') and (self.Aop_fp64 != NULL):
            return self.Aop_fp64.get_num_parameters()
        else:
            raise ValueError('Linear operator is not set.')

    # ==================
    # get data type name
    # ==================

    def get_data_type_name(self):
        """
        """

        if self.data_type_name == "":
            raise RuntimeError('Linear operator data type is not set.')

        return self.data_type_name

    # ============================
    # get linear operator fp8 e5m2
    # ============================

    cdef cuLinearOperator[__nv_fp8_e5m2]* get_linear_operator_fp8_e5m2(
            self) except *:
        """
        """

        if self.Aop_fp8_e5m2 == NULL:
            raise RuntimeError('Linear operator (__nv_fp8_e5m2 type) is not '
                               'set.')

        if self.data_type_name != b'float16':
            raise RuntimeError('Wrong accessors is called. The type of the ' +
                               'LinearOperator object is: %s'
                               % self.data_type_name)

        return self.Aop_fp8_e5m2

    # ============================
    # get linear operator fp8 e4m3
    # ============================

    cdef cuLinearOperator[__nv_fp8_e4m3]* get_linear_operator_fp8_e4m3(
            self) except *:
        """
        """

        if self.Aop_fp8_e4m3 == NULL:
            raise RuntimeError('Linear operator (__nv_fp8_e4m3 type) is not '
                               'set.')

        if self.data_type_name != b'float16':
            raise RuntimeError('Wrong accessors is called. The type of the ' +
                               'LinearOperator object is: %s'
                               % self.data_type_name)

        return self.Aop_fp8_e4m3

    # ========================
    # get linear operator fp16
    # ========================

    cdef cuLinearOperator[__half]* get_linear_operator_fp16(self) except *:
        """
        """

        if self.Aop_fp16 == NULL:
            raise RuntimeError('Linear operator (__half type) is not set.')

        if self.data_type_name != b'float16':
            raise RuntimeError('Wrong accessors is called. The type of the ' +
                               'LinearOperator object is: %s'
                               % self.data_type_name)

        return self.Aop_fp16

    # ========================
    # get linear operator bf16
    # ========================

    cdef cuLinearOperator[__nv_bfloat16]* get_linear_operator_bf16(
            self) except *:
        """
        """

        if self.Aop_bf16 == NULL:
            raise RuntimeError('Linear operator (__nv_bfloat16 type) is not '
                               'set.')

        if self.data_type_name != b'bfloat16':
            raise RuntimeError('Wrong accessors is called. The type of the ' +
                               'LinearOperator object is: %s'
                               % self.data_type_name)

        return self.Aop_bf16

    # ========================
    # get linear operator fp32
    # ========================

    cdef cuLinearOperator[float]* get_linear_operator_fp32(self) except *:
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

    cdef cuLinearOperator[double]* get_linear_operator_fp64(self) except *:
        """
        """

        if self.Aop_fp64 == NULL:
            raise RuntimeError('Linear operator (double type) is not set.')

        if self.data_type_name != b'float64':
            raise RuntimeError('Wrong accessors is called. The type of the ' +
                               'LinearOperator object is: %s'
                               % self.data_type_name)

        return self.Aop_fp64

    # =====================
    # get device properties
    # =====================

    def get_device_properties(self):
        """
        """

        if self.device_properties_dict is None:
            self.device_properties_dict = py_query_device()
        return self.device_properties_dict
    
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

        if (self.data_type_name == b'float8_e5m2') and \
                (self.Aop_fp8_e5m2 != NULL):
            self.Aop_fp8_e5m2.set_symmetry(symmetric)
        elif (self.data_type_name == b'float8_e4m3') and \
                (self.Aop_fp8_e4m3 != NULL):
            self.Aop_fp8_e4m3.set_symmetry(symmetric)
        elif (self.data_type_name == b'float16') and (self.Aop_fp16 != NULL):
            self.Aop_fp16.set_symmetry(symmetric)
        elif (self.data_type_name == b'bfloat16') and (self.Aop_bf16 != NULL):
            self.Aop_bf16.set_symmetry(symmetric)
        elif (self.data_type_name == b'float32') and (self.Aop_fp32 != NULL):
            self.Aop_fp32.set_symmetry(symmetric)    
        elif (self.data_type_name == b'float64') and (self.Aop_fp64 != NULL):
            self.Aop_fp64.set_symmetry(symmetric)
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

        # Declare c pointers for parameters
        cdef float* c_parameters_fp32
        cdef double* c_parameters_fp64

        if (self.data_type_name == b'float32') and (self.Aop_fp32 != NULL):
            mv_parameters_fp32 = self.parameters.astype('float32')
            c_parameters_fp32 = &mv_parameters_fp32[0]
            self.Aop_fp32.set_parameters(c_parameters_fp32)
            
        elif (self.data_type_name == b'float64') and (self.Aop_fp64 != NULL):
            mv_parameters_fp64 = self.parameters.astype('float64')
            c_parameters_fp64 = &mv_parameters_fp64[0]
            self.Aop_fp64.set_parameters(c_parameters_fp64)

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

        # Declare memory views for output product
        cdef float[:] mv_product_fp32
        cdef double[:] mv_product_fp64

        # Declare c pointers for input vector
        cdef float* c_vector_fp32
        cdef double* c_vector_fp64

        # Declare c pointers for output product
        cdef float* c_product_fp32
        cdef double* c_product_fp64

        # Dispatch to single or double precision
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

        else:
            raise TypeError('Vector type should be either "float32", or ' +
                            '"float64".')

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

        # Declare memory views for output product
        cdef float[:] mv_product_fp32
        cdef double[:] mv_product_fp64

        # Declare c pointers for input vector
        cdef float* c_vector_fp32
        cdef double* c_vector_fp64

        # Declare c pointers for output product
        cdef float* c_product_fp32
        cdef double* c_product_fp64

        # Dispatch to single or double precision
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

        else:
            raise TypeError('Vector type should be either "float32", or ' +
                            '"float64".')
