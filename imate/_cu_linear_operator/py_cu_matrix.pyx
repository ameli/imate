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

# Python
import numpy
from scipy.sparse import issparse, csr_matrix

# Cython
from .py_cu_linear_operator cimport pycuLinearOperator
from .cu_matrix cimport cuMatrix
from .cu_dense_matrix cimport cuDenseMatrix
from .cu_csr_matrix cimport cuCSRMatrix
from .cu_csc_matrix cimport cuCSCMatrix
from .._definitions.types cimport LongIndexType, FlagType
from .._array cimport get_array_buffer
from .._array import get_data_type_name, get_device, get_ndim, get_shape, \
        is_row_major
from .._cu_definitions.cu_types cimport __nv_fp8_e5m2, __nv_fp8_e4m3, __half, \
        __nv_bfloat16
from .._array cimport release_buffer


# ==========
# pycuMatrix
# ==========

cdef class pycuMatrix(pycuLinearOperator):
    """
    Defines a linear operator that is a constant matrix.

    **Initializing Object:**

    The object is initialized by a given matrix :math:`\\mathbf{A}` which can
    be a numpy array, or sparse matrices of any format (CSR, CSC, etc) using
    scipy sparse module.

    .. note::

        Initializing the linear operator requires python's GIL. Also, the
        following examples should be used in a ``*.pyx`` file and should be
        compiled as cython's extension module.

    In the following example, we create the object ``Aop`` based on
    scipy.sparse matrix of CSR format. Note the format of the input matrix
    can also be anything other than ``'csr'``, such as ``'csc'``.

    .. code-block:: python

        >>> # Use this script in a *.pyx file
        >>> import scipy.sparse

        >>> # Create to random sparse matrices
        >>> n, m = 1000
        >>> A = scipy.sparse.random(n, m, format='csr')

        >>> # Create linear operator object
        >>> from imate.linear_operator cimport ConstantMatrix
        >>> cdef ConstantMatrix Aop = ConstantMatrix(A)


    The following is an example of defining the operator with a dense matrix:

    .. code-block:: python

        >>> # Use this script in a *.pyx file
        >>> import numpy

        >>> # Create to random sparse matrices
        >>> n, m = 1000
        >>> cdef ConstantMatrix A = numpy.random.randn((n, m), dtype=float)

        >>> # Create linear operator object
        >>> from imate.linear_operator cimport ConstantMatrix
        >>> cdef ConstantMatrix Aop = ConstantMatrix(A)

    **Matrix-Vector Multiplications:**

    The linear operator can perform matrix vector multiplication using
    :func:`dot` function and the matrix-vector multiplication with the
    transposed matrix using :func:`transpose_dot` function.

    .. note::

        Matrix-vector multiplication using :func:`dot` and
        :func:`transpose_dot` functions do not require python's GIL, hence,
        they can be called in a ``nogil`` environment, if desired.

    .. code-block:: python

        >>> # Use this script in a *.pyx file
        >>> # Create a vectors as cython's memoryview to numpy arrays
        >>> import numpy
        >>> cdef double[:] b = numpy.random.randn(m)
        >>> cdef double[:] c = numpy.empty((n, 1), dtype=float)

        >>> # Perform product on vector b and store the product on vector c
        >>> with nogil:
        ...     Aop.dot(&b[0], &c[0])

        >>> # Perform product using the transpose of the operator
        >>> with nogil:
        >>>     Aop.transpose_dot(&b[0], &c[0])

        .. seealso::

            :class:`AffineMatrixFunction`
    """

    # =========
    # __cinit__
    # =========

    def __cinit__(self, A, A_is_symmetric=False, num_gpu_devices=0):
        """
        Sets the matrix A.
        """

        # Number of gpu devices to use. This might be different (less) than the
        # number of gpu devices that are available. If set to 0, all available
        # devices will be used.
        self.num_gpu_devices = num_gpu_devices

        # Check A
        if A is None:
            raise ValueError('A cannot be None.')

        if get_ndim(A) != 2:
            raise ValueError('Input matrix should be a 2-dimensional array.')

        # Symmetric matrix A
        if not isinstance(A_is_symmetric, bool):
            raise ValueError('"A_is_symmetric" should be boolean.')
        self.A_is_symmetric = A_is_symmetric

        # Data type
        self.data_type_name = get_data_type_name(A)

        if self.data_type_name not in [b'float8_e5m2', b'float8_e4m3',
                                       b'float16', b'bfloat16', b'float32',
                                       b'float64']:
            raise TypeError('When the computation is performed on GPU, the '
                            'data type should be either "float8_e5m2", '
                            '"float8_e4m3", "float16", "bfloat16", "float32", '
                            'or "float64".')

        # Determine A is sparse or dense
        if issparse(A):

            # Matrix type codes: 'r' for CSR, and 'c' for CSC
            if A.format == 'csr':

                # Check sorted indices
                if not A.has_sorted_indices:
                    A.sort_indices()

                # set CSR matrix
                self.set_csr_matrix(A)

            elif A.format == 'csc':

                # Check sorted indices
                if not A.has_sorted_indices:
                    A.sort_indices()

                # set CSC matrix
                self.set_csc_matrix(A)

            else:

                # If A is neither CSR or CSC, convert A to CSR
                self.A_csr = csr_matrix(A, dtype=A.dtype)

                # Check sorted indices
                if not self.A_csr.has_sorted_indices:
                    self.A_csr.sort_indices()

                # set CSR matrix
                self.set_csr_matrix(self.A_csr)

        else:
            # A is dense matrix
            self.set_dense_matrix(A)
            
    # ===========
    # __dealloc__
    # ===========

    def __dealloc__(self):
        """
        """

        # Release data buffer
        release_buffer(&self.A_data_py_buffer)
        release_buffer(&self.A_indices_py_buffer)
        release_buffer(&self.A_index_pointer_py_buffer)

    # ================
    # set dense matrix
    # ================

    def set_dense_matrix(self, A):
        """
        Sets matrix A.

        :param A: A 2-dimensional matrix.
        :type A: numpy.ndarray, or any scipy.sparse array
        """

        # Get shape
        num_rows, num_columns = get_shape(A)

        # Matrix size
        cdef LongIndexType A_num_rows = num_rows
        cdef LongIndexType A_num_columns = num_columns

        # Contiguity
        cdef FlagType A_is_row_major = is_row_major(A)

        # Declare pointer of A.data
        cdef const void* A_data = get_array_buffer(A, &self.A_data_py_buffer)

        # Create a linear operator object 
        if self.data_type_name == b'float8_e5m2':
            self.Aop_fp8_e5m2 = new cuDenseMatrix[__nv_fp8_e5m2](
                    <__nv_fp8_e5m2*> A_data,
                    A_num_rows,
                    A_num_columns,
                    A_is_row_major,
                    self.A_is_symmetric,
                    self.num_gpu_devices)

        elif self.data_type_name == b'float8_e4m3':
            self.Aop_fp8_e4m3 = new cuDenseMatrix[__nv_fp8_e4m3](
                    <__nv_fp8_e4m3*> A_data,
                    A_num_rows,
                    A_num_columns,
                    A_is_row_major,
                    self.A_is_symmetric,
                    self.num_gpu_devices)

        if self.data_type_name == b'float16':
            self.Aop_fp16 = new cuDenseMatrix[__half](
                    <__half*> A_data,
                    A_num_rows,
                    A_num_columns,
                    A_is_row_major,
                    self.A_is_symmetric,
                    self.num_gpu_devices)

        elif self.data_type_name == b'bfloat16':
            self.Aop_bf16 = new cuDenseMatrix[__nv_bfloat16](
                    <__nv_bfloat16*> A_data,
                    A_num_rows,
                    A_num_columns,
                    A_is_row_major,
                    self.A_is_symmetric,
                    self.num_gpu_devices)

        elif self.data_type_name == b'float32':
            self.Aop_fp32 = new cuDenseMatrix[float](
                    <float*> A_data,
                    A_num_rows,
                    A_num_columns,
                    A_is_row_major,
                    self.A_is_symmetric,
                    self.num_gpu_devices)

        elif self.data_type_name == b'float64':
            self.Aop_fp64 = new cuDenseMatrix[double](
                    <double*> A_data,
                    A_num_rows,
                    A_num_columns,
                    A_is_row_major,
                    self.A_is_symmetric,
                    self.num_gpu_devices)

    # ==============
    # set csr matrix
    # ==============

    def set_csr_matrix(self, A):
        """
        """

        # Get shape
        num_rows, num_columns = get_shape(A)

        # Matrix size
        cdef LongIndexType A_num_rows = num_rows
        cdef LongIndexType A_num_columns = num_columns

        # Declare pointer for A.data
        cdef const void* A_data = get_array_buffer(
                A.data, &self.A_data_py_buffer)

        # If the input type is the same as LongIndexType, no copy is performed.
        self.A_indices_copy = \
            A.indices.astype(self.long_index_type_name, copy=False)
        self.A_index_pointer_copy = \
            A.indptr.astype(self.long_index_type_name, copy=False)

        # Declare pointers to A.indices ans A.indptr
        cdef const void* A_indices = get_array_buffer(
                self.A_indices_copy, &self.A_indices_py_buffer)
        cdef const void* A_index_pointer = get_array_buffer(
                self.A_index_pointer_copy, &self.A_index_pointer_py_buffer)

        # Create a linear operator object
        if self.data_type_name == b'float8_e5m2':
            self.Aop_fp8_e5m2 = new cuCSRMatrix[__nv_fp8_e5m2](
                    <__nv_fp8_e5m2*> A_data,
                    <LongIndexType*> A_indices,
                    <LongIndexType*> A_index_pointer,
                    A_num_rows,
                    A_num_columns,
                    self.A_is_symmetric,
                    self.num_gpu_devices)

        elif self.data_type_name == b'float8_e4m3':
            self.Aop_fp8_e4m3 = new cuCSRMatrix[__nv_fp8_e4m3](
                    <__nv_fp8_e4m3*> A_data,
                    <LongIndexType*> A_indices,
                    <LongIndexType*> A_index_pointer,
                    A_num_rows,
                    A_num_columns,
                    self.A_is_symmetric,
                    self.num_gpu_devices)

        elif self.data_type_name == b'float16':
            self.Aop_fp16 = new cuCSRMatrix[__half](
                    <__half*> A_data,
                    <LongIndexType*> A_indices,
                    <LongIndexType*> A_index_pointer,
                    A_num_rows,
                    A_num_columns,
                    self.A_is_symmetric,
                    self.num_gpu_devices)

        elif self.data_type_name == b'bfloat16':
            self.Aop_bf16 = new cuCSRMatrix[__nv_bfloat16](
                    <__nv_bfloat16*> A_data,
                    <LongIndexType*> A_indices,
                    <LongIndexType*> A_index_pointer,
                    A_num_rows,
                    A_num_columns,
                    self.A_is_symmetric,
                    self.num_gpu_devices)

        elif self.data_type_name == b'float32':
            self.Aop_fp32 = new cuCSRMatrix[float](
                    <float*> A_data,
                    <LongIndexType*> A_indices,
                    <LongIndexType*> A_index_pointer,
                    A_num_rows,
                    A_num_columns,
                    self.A_is_symmetric,
                    self.num_gpu_devices)

        elif self.data_type_name == b'float64':
            self.Aop_fp64 = new cuCSRMatrix[double](
                    <double*> A_data,
                    <LongIndexType*> A_indices,
                    <LongIndexType*> A_index_pointer,
                    A_num_rows,
                    A_num_columns,
                    self.A_is_symmetric,
                    self.num_gpu_devices)

    # ==============
    # set csc matrix
    # ==============

    def set_csc_matrix(self, A):
        """
        """

        # Get shape
        num_rows, num_columns = get_shape(A)

        # Matrix size
        cdef LongIndexType A_num_rows = num_rows
        cdef LongIndexType A_num_columns = num_columns

        # Declare pointer for A.data
        cdef const void* A_data = get_array_buffer(
                A.data, &self.A_data_py_buffer)

        # If the input type is the same as LongIndexType, no copy is performed.
        self.A_indices_copy = \
            A.indices.astype(self.long_index_type_name, copy=False)
        self.A_index_pointer_copy = \
            A.indptr.astype(self.long_index_type_name, copy=False)

        # Declare pointers to A.indices ans A.indptr
        cdef const void* A_indices = get_array_buffer(
                self.A_indices_copy, &self.A_indices_py_buffer)
        cdef const void* A_index_pointer = get_array_buffer(
                self.A_index_pointer_copy, &self.A_index_pointer_py_buffer)

        # Create a linear operator object
        if self.data_type_name == b'float8_e5m2':
            self.Aop_fp8_e5m2 = new cuCSCMatrix[__nv_fp8_e5m2](
                    <__nv_fp8_e5m2*> A_data,
                    <LongIndexType*> A_indices,
                    <LongIndexType*> A_index_pointer,
                    A_num_rows,
                    A_num_columns,
                    self.A_is_symmetric,
                    self.num_gpu_devices)

        elif self.data_type_name == b'float8_e4m3':
            self.Aop_fp8_e4m3 = new cuCSCMatrix[__nv_fp8_e4m3](
                    <__nv_fp8_e4m3*> A_data,
                    <LongIndexType*> A_indices,
                    <LongIndexType*> A_index_pointer,
                    A_num_rows,
                    A_num_columns,
                    self.A_is_symmetric,
                    self.num_gpu_devices)

        elif self.data_type_name == b'float16':
            self.Aop_fp16 = new cuCSCMatrix[__half](
                    <__half*> A_data,
                    <LongIndexType*> A_indices,
                    <LongIndexType*> A_index_pointer,
                    A_num_rows,
                    A_num_columns,
                    self.A_is_symmetric,
                    self.num_gpu_devices)

        elif self.data_type_name == b'bfloat16':
            self.Aop_bf16 = new cuCSCMatrix[__nv_bfloat16](
                    <__nv_bfloat16*> A_data,
                    <LongIndexType*> A_indices,
                    <LongIndexType*> A_index_pointer,
                    A_num_rows,
                    A_num_columns,
                    self.A_is_symmetric,
                    self.num_gpu_devices)

        elif self.data_type_name == b'float32':
            self.Aop_fp32 = new cuCSCMatrix[float](
                    <float*> A_data,
                    <LongIndexType*> A_indices,
                    <LongIndexType*> A_index_pointer,
                    A_num_rows,
                    A_num_columns,
                    self.A_is_symmetric,
                    self.num_gpu_devices)

        elif self.data_type_name == b'float64':
            self.Aop_fp64 = new cuCSCMatrix[double](
                    <double*> A_data,
                    <LongIndexType*> A_indices,
                    <LongIndexType*> A_index_pointer,
                    A_num_rows,
                    A_num_columns,
                    self.A_is_symmetric,
                    self.num_gpu_devices)
