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
from .cu_dense_affine_matrix_function cimport cuDenseAffineMatrixFunction
from .cu_csr_affine_matrix_function cimport cuCSRAffineMatrixFunction
from .cu_csc_affine_matrix_function cimport cuCSCAffineMatrixFunction
from .._definitions.types cimport LongIndexType, FlagType
from .._array cimport get_array_buffer
from .._array import get_data_type_name, get_device, get_shape, get_ndim, \
        is_row_major
from .._cu_definitions.cu_types cimport __nv_fp8_e5m2, __nv_fp8_e4m3, __half, \
        __nv_bfloat16
from .._array cimport release_buffer


# ===========================
# pycu Affine Matrix Function
# ===========================

cdef class pycuAffineMatrixFunction(pycuLinearOperator):
    """
    Defines a linear operator that is an affine function of a single parameter.
    Given two matrices :math:`\\mathbf{A}` and :math:`\\mathbf{B}`, the linear
    operator is defined by

    .. math::

        \\mathbf{A}(t) = \\mathbf{A} + t \\mathbf{B},

    where :math:`t \\in \\mathbb{R}` is a parameter.

    **Initializing Object:**

    The matrices :math:`\\mathbf{A}` and :math:`\\mathbf{B}` are given at the
    initialization of the object. These matrices can be a dense matrix as 2D
    numpy arrays, or sparse matrices of any format (CSR, CSC, etc) using scipy
    sparse module.

    .. note::

        Initializing the linear operator requires python's GIL. Also, the
        following examples should be used in a ``*.pyx`` file and should be
        compiled as cython's extension module.

    In the following example, we create the object ``Aop`` based on
    scipy.sparse matrices of CSR format. Note the format of the input matrices
    can also be anything other than ``'csr'``, such as ``'csc'``.

    .. code-block:: python

        >>> # Use this script in a *.pyx file
        >>> import scipy.sparse

        >>> # Create to random sparse matrices
        >>> n, m = 1000
        >>> A = scipy.sparse.random(n, m, format='csr')
        >>> B = scipy.sparse.random(n, m, format='csr')

        >>> # Create linear operator object
        >>> from imate.linear_operator cimport AffineMatrixFunction
        >>> cdef AffineMatrixFunction Aop = AffineMatrixFunction(A, B)


    The following is an example of defining the operator with dense matrices:

    .. code-block:: python

        >>> # Use this script in a *.pyx file
        >>> import numpy

        >>> # Create to random sparse matrices
        >>> n, m = 1000
        >>> A = numpy.random.randn((n, m), dtype=float)
        >>> B = numpy.random.randn((n, m), dtype=float)

        >>> # Create linear operator object
        >>> from imate.linear_operator cimport AffineMatrixFunction
        >>> cdef AffineMatrixFunction Aop = AffineMatrixFunction(A, B)

    If the matrix ``B`` is not given, or if it is ``None``, or if it is ``0``,
    then the linear operator assumes ``B`` is zero matrix. For example:

    .. code-block:: python

        # Case 1: Not providing B
        >>> cdef AffineMatrixFunction Aop = AffineMatrixFunction(A)

        # Case 2: Setting B to None
        >>> cdef AffineMatrixFunction Aop = AffineMatrixFunction(A, None)

        # Case 3: Setting B to scalar zero
        >>> cdef AffineMatrixFunction Aop = AffineMatrixFunction(A, 0)

    If the matrix ``B`` is set to the scalar ``1``, the linear operator assumes
    that ``B`` is the identity matrix. For example:

    .. code-block:: python

        >>> cdef AffineMatrixFunction Aop = AffineMatrixFunction(A, 1)

    **Setting the Parameter:**

    The parameter :math:`t` is given to the object ``Aop`` at **runtime** using
    :func:`set_parameters` function.

    .. note::

        Setting the parameter using :func:`set_parameter` does not require
        python's GIL, hence, the parameter can be set in ``nogil`` environment,
        if desired.

    .. code-block:: python

        >>> # Use this script in a *.pyx file
        >>> cdef double t = 1.0

        >>> # nogil environment is optional
        >>> with nogil:
        ...     Aop.set_parameters(&t)

    Note that a *pointer* to the parameter should be provided to the function.

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

            :class:`Matrix`
    """

    # =========
    # __cinit__
    # =========

    def __cinit__(
            self,
            A,
            B=None,
            A_is_symmetric=False,
            B_is_symmetric=False,
            num_gpu_devices=0):
        """
        Sets matrices A and B.
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

        # Data type
        self.data_type_name = get_data_type_name(A)

        if self.data_type_name not in [b'float8_e5m2', b'float8_e4m3',
                                       b'float16', b'bfloat16', b'float32',
                                       b'float64']:
            raise TypeError('When the computation is performed on GPU, the '
                            'data type should be either "float8_e5m2", '
                            '"float8_e4m3", "float16", "bfloat16", "float32", '
                            'or "float64".')

        # Symmetric matrix A
        if not isinstance(A_is_symmetric, bool):
            raise ValueError('"A_is_symmetric" should be boolean.')
        self.A_is_symmetric = A_is_symmetric

        # Check if B is not to be considered as identity matrix
        if B is None:

            # B is assumed to be identity
            B_is_identity = True
            self.B_is_symmetric = True

        else:

            # B is neither zero nor identity
            B_is_identity = False

            # Symmetric matrix B
            if not isinstance(B_is_symmetric, bool):
                raise ValueError('"B_is_symmetric" should be boolean.')
            self.B_is_symmetric = B_is_symmetric

            # Check similar types of A and B
            if issparse(A) and issparse(B):
                if (A.format != B.format):
                    raise TypeError('Sparse matrices A and B should have '
                                    'similar formats.')
            elif not (type(A) == type(B)):
                raise TypeError('Matrices A and B should have similar types.')

            # Check A and B have the same data types
            if not (A.dtype == B.dtype):
                raise TypeError('A and B should have similar data types.')

            # Check consistent sizes of A and B
            if not (A.shape == B.shape):
                raise ValueError('A and B should have the same shape.')

        # Determine A is sparse or dense
        if issparse(A):

            # Matrix type codes: 'r' for CSR, and 'c' for CSC
            if A.format == 'csr':

                # Check sorted indices
                if not A.has_sorted_indices:
                    A.sort_indices()

                if (not B_is_identity) and (not B.has_sorted_indices):
                    B.sort_indices()

                # set CSR matrix
                self.set_csr_matrix(A, B, B_is_identity)

            elif A.format == 'csc':

                # Check sorted indices
                if not A.has_sorted_indices:
                    A.sort_indices()

                if (not B_is_identity) and (not B.has_sorted_indices):
                    B.sort_indices()

                # set CSC matrix
                self.set_csc_matrix(A, B, B_is_identity)

            else:

                # If A is neither CSR or CSC, convert A to CSR
                self.A_csr = csr_matrix(A, dtype=A.dtype)

                if not B_is_identity:
                    self.B_csr = csr_matrix(B, dtype=B.dtype)
                else:
                    self.B_csr = B

                # Check sorted indices
                if not self.A_csr.has_sorted_indices:
                    self.A_csr.sort_indices()

                if (not B_is_identity) and (not self.B_csr.has_sorted_indices):
                    self.B_csr.sort_indices()

                # set CSR matrix
                self.set_csr_matrix(self.A_csr, self.B_csr, B_is_identity)

        else:
            # A and B are dense matrices
            self.set_dense_matrix(A, B, B_is_identity)
    
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
        release_buffer(&self.B_data_py_buffer)
        release_buffer(&self.B_indices_py_buffer)
        release_buffer(&self.B_index_pointer_py_buffer)

    # ================
    # set dense matrix
    # ================

    def set_dense_matrix(self, A, B, B_is_identity):
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
        cdef FlagType B_is_row_major = 0

        if not B_is_identity:
            B_is_row_major = is_row_major(B)

        # Declare pointer of A.data and B.data
        cdef const void* A_data = get_array_buffer(A, &self.A_data_py_buffer)
        cdef const void* B_data

        # Get pointer to data of B depending on row or column major
        if B_is_identity:
            B_data = NULL
        else:
            B_data = get_array_buffer(B, &self.B_data_py_buffer)

        # Create a linear operator object
        if self.data_type_name == b'float8_e5m2':

            if B_is_identity:
                IF USE_CUDA_FP8_E5M2:
                    self.Aop_fp8_e5m2 = \
                        new cuDenseAffineMatrixFunction[__nv_fp8_e5m2](
                            <__nv_fp8_e5m2*> A_data,
                            A_num_rows,
                            A_num_columns,
                            A_is_row_major,
                            self.A_is_symmetric,
                            self.num_gpu_devices)
                ELSE:
                    raise TypeError('Package was not compiled with FP8-E5M2 '
                                    'precision support for CUDA.')
            else:
                IF USE_CUDA_FP8_E5M2:
                    self.Aop_fp8_e5m2 = \
                        new cuDenseAffineMatrixFunction[__nv_fp8_e5m2](
                            <__nv_fp8_e5m2*> A_data,
                            A_num_rows,
                            A_num_columns,
                            A_is_row_major,
                            self.A_is_symmetric,
                            <__nv_fp8_e5m2*> B_data,
                            B_is_row_major,
                            self.B_is_symmetric,
                            self.num_gpu_devices)
                ELSE:
                    raise TypeError('Package was not compiled with FP8-E5M2 '
                                    'precision support for CUDA.')

        elif self.data_type_name == b'float8_e4m3':

            if B_is_identity:
                IF USE_CUDA_FP8_E4M3:
                    self.Aop_fp8_e4m3 = \
                        new cuDenseAffineMatrixFunction[__nv_fp8_e4m3](
                            <__nv_fp8_e4m3*> A_data,
                            A_num_rows,
                            A_num_columns,
                            A_is_row_major,
                            self.A_is_symmetric,
                            self.num_gpu_devices)
                ELSE:
                    raise TypeError('Package was not compiled with FP8-E4M3 '
                                    'precision support for CUDA.')
            else:
                IF USE_CUDA_FP8_E4M3:
                    self.Aop_fp8_e4m3 = \
                        new cuDenseAffineMatrixFunction[__nv_fp8_e4m3](
                            <__nv_fp8_e4m3*> A_data,
                            A_num_rows,
                            A_num_columns,
                            A_is_row_major,
                            self.A_is_symmetric,
                            <__nv_fp8_e4m3*> B_data,
                            B_is_row_major,
                            self.B_is_symmetric,
                            self.num_gpu_devices)
                ELSE:
                    raise TypeError('Package was not compiled with FP8-E4M3 '
                                    'precision support for CUDA.')

        elif self.data_type_name == b'bfloat16':

            if B_is_identity:
                IF USE_CUDA_BF16:
                    self.Aop_bf16 = \
                        new cuDenseAffineMatrixFunction[__nv_bfloat16](
                            <__nv_bfloat16*> A_data,
                            A_num_rows,
                            A_num_columns,
                            A_is_row_major,
                            self.A_is_symmetric,
                            self.num_gpu_devices)
                ELSE:
                    raise TypeError('Package was not compiled with BF16 '
                                    'precision support for CUDA.')
            else:
                IF USE_CUDA_BF16:
                    self.Aop_bf16 = \
                        new cuDenseAffineMatrixFunction[__nv_bfloat16](
                            <__nv_bfloat16*> A_data,
                            A_num_rows,
                            A_num_columns,
                            A_is_row_major,
                            self.A_is_symmetric,
                            <__nv_bfloat16*> B_data,
                            B_is_row_major,
                            self.B_is_symmetric,
                            self.num_gpu_devices)
                ELSE:
                    raise TypeError('Package was not compiled with BF16 '
                                    'precision support for CUDA.')

        elif self.data_type_name == b'float16':

            if B_is_identity:
                IF USE_CUDA_FP16:
                    self.Aop_fp16 = new cuDenseAffineMatrixFunction[__half](
                            <__half*> A_data,
                            A_num_rows,
                            A_num_columns,
                            A_is_row_major,
                            self.A_is_symmetric,
                            self.num_gpu_devices)
                ELSE:
                    raise TypeError('Package was not compiled with FP16 '
                                    'precision support for CUDA.')
            else:
                IF USE_CUDA_FP16:
                    self.Aop_fp16 = new cuDenseAffineMatrixFunction[__half](
                            <__half*> A_data,
                            A_num_rows,
                            A_num_columns,
                            A_is_row_major,
                            self.A_is_symmetric,
                            <__half*> B_data,
                            B_is_row_major,
                            self.B_is_symmetric,
                            self.num_gpu_devices)
                ELSE:
                    raise TypeError('Package was not compiled with FP16 '
                                    'precision support for CUDA.')

        elif self.data_type_name == b'float32':

            if B_is_identity:
                IF USE_CUDA_FP32:
                    self.Aop_fp32 = new cuDenseAffineMatrixFunction[float](
                            <float*> A_data,
                            A_num_rows,
                            A_num_columns,
                            A_is_row_major,
                            self.A_is_symmetric,
                            self.num_gpu_devices)
                ELSE:
                    raise TypeError('Package was not compiled with FP32 '
                                    'precision support for CUDA.')
            else:
                IF USE_CUDA_FP32:
                    self.Aop_fp32 = new cuDenseAffineMatrixFunction[float](
                            <float*> A_data,
                            A_num_rows,
                            A_num_columns,
                            A_is_row_major,
                            self.A_is_symmetric,
                            <float*> B_data,
                            B_is_row_major,
                            self.B_is_symmetric,
                            self.num_gpu_devices)
                ELSE:
                    raise TypeError('Package was not compiled with FP32 '
                                    'precision support for CUDA.')

        elif self.data_type_name == b'float64':

            if B_is_identity:
                IF USE_CUDA_FP64:
                    self.Aop_fp64 = new cuDenseAffineMatrixFunction[double](
                            <double*> A_data,
                            A_num_rows,
                            A_num_columns,
                            A_is_row_major,
                            self.A_is_symmetric,
                            self.num_gpu_devices)
                ELSE:
                    raise TypeError('Package was not compiled with FP64 '
                                    'precision support for CUDA.')
            else:
                IF USE_CUDA_FP64:
                    self.Aop_fp64 = new cuDenseAffineMatrixFunction[double](
                            <double*> A_data,
                            A_num_rows,
                            A_num_columns,
                            A_is_row_major,
                            self.A_is_symmetric,
                            <double*> B_data,
                            B_is_row_major,
                            self.B_is_symmetric,
                            self.num_gpu_devices)
                ELSE:
                    raise TypeError('Package was not compiled with FP64 '
                                    'precision support for CUDA.')

    # ==============
    # set csr matrix
    # ==============

    def set_csr_matrix(self, A, B, B_is_identity):
        """
        """

        # Get shape
        num_rows, num_columns = get_shape(A)

        # Matrix size
        cdef LongIndexType A_num_rows = num_rows
        cdef LongIndexType A_num_columns = num_columns

        # If the input type is the same as LongIndexType, no copy is performed.
        self.A_indices_copy = \
            A.indices.astype(self.long_index_type_name, copy=False)
        self.A_index_pointer_copy = \
            A.indptr.astype(self.long_index_type_name, copy=False)

        if not B_is_identity:

            # If input type is the same as LongIndexType, no copy is performed.
            self.B_indices_copy = \
                B.indices.astype(self.long_index_type_name, copy=False)
            self.B_index_pointer_copy = \
                B.indptr.astype(self.long_index_type_name, copy=False)

        # Declare pointers
        cdef const void* A_data = get_array_buffer(
                A.data, &self.A_data_py_buffer)
        cdef const void* A_indices = get_array_buffer(
                self.A_indices_copy, &self.A_indices_py_buffer)
        cdef const void* A_index_pointer = get_array_buffer(
                self.A_index_pointer_copy, &self.A_index_pointer_py_buffer)
        cdef const void* B_data
        cdef const void* B_indices
        cdef const void* B_index_pointer

        if B_is_identity:
            B_data = NULL
            B_indices = NULL
            B_index_pointer = NULL
        else:
            B_data = get_array_buffer(B.data, &self.B_data_py_buffer)
            B_indices = get_array_buffer(
                    self.B_indices_copy, &self.B_indices_py_buffer)
            B_index_pointer = get_array_buffer(
                    self.B_index_pointer_copy, &self.B_index_pointer_py_buffer)

        # Create a linear operator object
        if self.data_type_name == b'float8_e5m2':

            if B_is_identity:
                IF USE_CUDA_FP8_E5M2:
                    self.Aop_fp8_e5m2 = \
                        new cuCSRAffineMatrixFunction[__nv_fp8_e5m2](
                            <__nv_fp8_e5m2*> A_data,
                            <LongIndexType*> A_indices,
                            <LongIndexType*> A_index_pointer,
                            A_num_rows,
                            A_num_columns,
                            self.A_is_symmetric,
                            self.num_gpu_devices)
                ELSE:
                    raise TypeError('Package was not compiled with FP8-E5M2 '
                                    'precision support for CUDA.')
            else:
                IF USE_CUDA_FP8_E5M2:
                    self.Aop_fp8_e5m2 = \
                        new cuCSRAffineMatrixFunction[__nv_fp8_e5m2](
                            <__nv_fp8_e5m2*> A_data,
                            <LongIndexType*> A_indices,
                            <LongIndexType*> A_index_pointer,
                            A_num_rows,
                            A_num_columns,
                            self.A_is_symmetric,
                            <__nv_fp8_e5m2*> B_data,
                            <LongIndexType*> B_indices,
                            <LongIndexType*> B_index_pointer,
                            self.B_is_symmetric,
                            self.num_gpu_devices)
                ELSE:
                    raise TypeError('Package was not compiled with FP8-E5M2 '
                                    'precision support for CUDA.')

        elif self.data_type_name == b'float8_e4m3':

            if B_is_identity:
                IF USE_CUDA_FP8_E4M3:
                    self.Aop_fp8_e4m3 = \
                        new cuCSRAffineMatrixFunction[__nv_fp8_e4m3](
                            <__nv_fp8_e4m3*> A_data,
                            <LongIndexType*> A_indices,
                            <LongIndexType*> A_index_pointer,
                            A_num_rows,
                            A_num_columns,
                            self.A_is_symmetric,
                            self.num_gpu_devices)
                ELSE:
                    raise TypeError('Package was not compiled with FP8-E4M3 '
                                    'precision support for CUDA.')
            else:
                IF USE_CUDA_FP8_E4M3:
                    self.Aop_fp8_e4m3 = \
                        new cuCSRAffineMatrixFunction[__nv_fp8_e4m3](
                            <__nv_fp8_e4m3*> A_data,
                            <LongIndexType*> A_indices,
                            <LongIndexType*> A_index_pointer,
                            A_num_rows,
                            A_num_columns,
                            self.A_is_symmetric,
                            <__nv_fp8_e4m3*> B_data,
                            <LongIndexType*> B_indices,
                            <LongIndexType*> B_index_pointer,
                            self.B_is_symmetric,
                            self.num_gpu_devices)
                ELSE:
                    raise TypeError('Package was not compiled with FP8-E4M3 '
                                    'precision support for CUDA.')

        elif self.data_type_name == b'bfloat16':

            if B_is_identity:
                IF USE_CUDA_BF16:
                    self.Aop_bf16 = \
                        new cuCSRAffineMatrixFunction[__nv_bfloat16](
                            <__nv_bfloat16*> A_data,
                            <LongIndexType*> A_indices,
                            <LongIndexType*> A_index_pointer,
                            A_num_rows,
                            A_num_columns,
                            self.A_is_symmetric,
                            self.num_gpu_devices)
                ELSE:
                    raise TypeError('Package was not compiled with BF16 '
                                    'precision support for CUDA.')
            else:
                IF USE_CUDA_BF16:
                    self.Aop_bf16 = \
                        new cuCSRAffineMatrixFunction[__nv_bfloat16](
                            <__nv_bfloat16*> A_data,
                            <LongIndexType*> A_indices,
                            <LongIndexType*> A_index_pointer,
                            A_num_rows,
                            A_num_columns,
                            self.A_is_symmetric,
                            <__nv_bfloat16*> B_data,
                            <LongIndexType*> B_indices,
                            <LongIndexType*> B_index_pointer,
                            self.B_is_symmetric,
                            self.num_gpu_devices)
                ELSE:
                    raise TypeError('Package was not compiled with BF16 '
                                    'precision support for CUDA.')

        elif self.data_type_name == b'float16':

            if B_is_identity:
                IF USE_CUDA_FP16:
                    self.Aop_fp16 = new cuCSRAffineMatrixFunction[__half](
                            <__half*> A_data,
                            <LongIndexType*> A_indices,
                            <LongIndexType*> A_index_pointer,
                            A_num_rows,
                            A_num_columns,
                            self.A_is_symmetric,
                            self.num_gpu_devices)
                ELSE:
                    raise TypeError('Package was not compiled with FP16 '
                                    'precision support for CUDA.')
            else:
                IF USE_CUDA_FP16:
                    self.Aop_fp16 = new cuCSRAffineMatrixFunction[__half](
                            <__half*> A_data,
                            <LongIndexType*> A_indices,
                            <LongIndexType*> A_index_pointer,
                            A_num_rows,
                            A_num_columns,
                            self.A_is_symmetric,
                            <__half*> B_data,
                            <LongIndexType*> B_indices,
                            <LongIndexType*> B_index_pointer,
                            self.B_is_symmetric,
                            self.num_gpu_devices)
                ELSE:
                    raise TypeError('Package was not compiled with FP16 '
                                    'precision support for CUDA.')

        elif self.data_type_name == b'float32':

            if B_is_identity:
                IF USE_CUDA_FP32:
                    self.Aop_fp32 = new cuCSRAffineMatrixFunction[float](
                            <float*> A_data,
                            <LongIndexType*> A_indices,
                            <LongIndexType*> A_index_pointer,
                            A_num_rows,
                            A_num_columns,
                            self.A_is_symmetric,
                            self.num_gpu_devices)
                ELSE:
                    raise TypeError('Package was not compiled with FP32 '
                                    'precision support for CUDA.')
            else:
                IF USE_CUDA_FP32:
                    self.Aop_fp32 = new cuCSRAffineMatrixFunction[float](
                            <float*> A_data,
                            <LongIndexType*> A_indices,
                            <LongIndexType*> A_index_pointer,
                            A_num_rows,
                            A_num_columns,
                            self.A_is_symmetric,
                            <float*> B_data,
                            <LongIndexType*> B_indices,
                            <LongIndexType*> B_index_pointer,
                            self.B_is_symmetric,
                            self.num_gpu_devices)
                ELSE:
                    raise TypeError('Package was not compiled with FP32 '
                                    'precision support for CUDA.')

        elif self.data_type_name == b'float64':

            if B_is_identity:
                IF USE_CUDA_FP64:
                    self.Aop_fp64 = new cuCSRAffineMatrixFunction[double](
                            <double*> A_data,
                            <LongIndexType*> A_indices,
                            <LongIndexType*> A_index_pointer,
                            A_num_rows,
                            A_num_columns,
                            self.A_is_symmetric,
                            self.num_gpu_devices)
                ELSE:
                    raise TypeError('Package was not compiled with FP64 '
                                    'precision support for CUDA.')
            else:
                IF USE_CUDA_FP64:
                    self.Aop_fp64 = new cuCSRAffineMatrixFunction[double](
                            <double*> A_data,
                            <LongIndexType*> A_indices,
                            <LongIndexType*> A_index_pointer,
                            A_num_rows,
                            A_num_columns,
                            self.A_is_symmetric,
                            <double*> B_data,
                            <LongIndexType*> B_indices,
                            <LongIndexType*> B_index_pointer,
                            self.B_is_symmetric,
                            self.num_gpu_devices)
                ELSE:
                    raise TypeError('Package was not compiled with FP64 '
                                    'precision support for CUDA.')

    # ==============
    # set csc matrix
    # ==============

    def set_csc_matrix(self, A, B, B_is_identity):
        """
        """

        # Get shape
        num_rows, num_columns = get_shape(A)

        # Matrix size
        cdef LongIndexType A_num_rows = num_rows
        cdef LongIndexType A_num_columns = num_columns

        # If the input type is the same as LongIndexType, no copy is performed.
        self.A_indices_copy = \
            A.indices.astype(self.long_index_type_name, copy=False)
        self.A_index_pointer_copy = \
            A.indptr.astype(self.long_index_type_name, copy=False)

        if not B_is_identity:

            # If input type is the same as LongIndexType, no copy is performed.
            self.B_indices_copy = \
                B.indices.astype(self.long_index_type_name, copy=False)
            self.B_index_pointer_copy = \
                B.indptr.astype(self.long_index_type_name, copy=False)

        # Declare pointers
        cdef const void* A_data = get_array_buffer(
                A.data, &self.A_data_py_buffer)
        cdef const void* A_indices = get_array_buffer(
                self.A_indices_copy, &self.A_indices_py_buffer)
        cdef const void* A_index_pointer = get_array_buffer(
                self.A_index_pointer_copy, &self.A_index_pointer_py_buffer)
        cdef const void* B_data
        cdef const void* B_indices
        cdef const void* B_index_pointer

        if B_is_identity:
            B_data = NULL
            B_indices = NULL
            B_index_pointer = NULL
        else:
            B_data = get_array_buffer(B.data, &self.B_data_py_buffer)
            B_indices = get_array_buffer(
                    self.B_indices_copy, &self.B_indices_py_buffer)
            B_index_pointer = get_array_buffer(
                    self.B_index_pointer_copy, &self.B_index_pointer_py_buffer)

        # Create a linear operator object
        if self.data_type_name == b'float8_e5m2':

            if B_is_identity:
                IF USE_CUDA_FP8_E5M2:
                    self.Aop_fp8_e5m2 = \
                        new cuCSCAffineMatrixFunction[__nv_fp8_e5m2](
                            <__nv_fp8_e5m2*> A_data,
                            <LongIndexType*> A_indices,
                            <LongIndexType*> A_index_pointer,
                            A_num_rows,
                            A_num_columns,
                            self.A_is_symmetric,
                            self.num_gpu_devices)
                ELSE:
                    raise TypeError('Package was not compiled with FP8-E5M2 '
                                    'precision support for CUDA.')
            else:
                IF USE_CUDA_FP8_E5M2:
                    self.Aop_fp8_e5m2 = \
                        new cuCSCAffineMatrixFunction[__nv_fp8_e5m2](
                            <__nv_fp8_e5m2*> A_data,
                            <LongIndexType*> A_indices,
                            <LongIndexType*> A_index_pointer,
                            A_num_rows,
                            A_num_columns,
                            self.A_is_symmetric,
                            <__nv_fp8_e5m2*> B_data,
                            <LongIndexType*> B_indices,
                            <LongIndexType*> B_index_pointer,
                            self.B_is_symmetric,
                            self.num_gpu_devices)
                ELSE:
                    raise TypeError('Package was not compiled with FP8-E5M2 '
                                    'precision support for CUDA.')

        elif self.data_type_name == b'float8_e4m3':

            if B_is_identity:
                IF USE_CUDA_FP8_E4M3:
                    self.Aop_fp8_e4m3 = \
                        new cuCSCAffineMatrixFunction[__nv_fp8_e4m3](
                            <__nv_fp8_e4m3*> A_data,
                            <LongIndexType*> A_indices,
                            <LongIndexType*> A_index_pointer,
                            A_num_rows,
                            A_num_columns,
                            self.A_is_symmetric,
                            self.num_gpu_devices)
                ELSE:
                    raise TypeError('Package was not compiled with FP8-E4M3 '
                                    'precision support for CUDA.')
            else:
                IF USE_CUDA_FP8_E4M3:
                    self.Aop_fp8_e4m3 = \
                        new cuCSCAffineMatrixFunction[__nv_fp8_e4m3](
                            <__nv_fp8_e4m3*> A_data,
                            <LongIndexType*> A_indices,
                            <LongIndexType*> A_index_pointer,
                            A_num_rows,
                            A_num_columns,
                            self.A_is_symmetric,
                            <__nv_fp8_e4m3*> B_data,
                            <LongIndexType*> B_indices,
                            <LongIndexType*> B_index_pointer,
                            self.B_is_symmetric,
                            self.num_gpu_devices)
                ELSE:
                    raise TypeError('Package was not compiled with FP8-E4M3 '
                                    'precision support for CUDA.')

        elif self.data_type_name == b'bfloat16':

            if B_is_identity:
                IF USE_CUDA_BF16:
                    self.Aop_bf16 = \
                        new cuCSCAffineMatrixFunction[__nv_bfloat16](
                            <__nv_bfloat16*> A_data,
                            <LongIndexType*> A_indices,
                            <LongIndexType*> A_index_pointer,
                            A_num_rows,
                            A_num_columns,
                            self.A_is_symmetric,
                            self.num_gpu_devices)
                ELSE:
                    raise TypeError('Package was not compiled with BF16 '
                                    'precision support for CUDA.')
            else:
                IF USE_CUDA_BF16:
                    self.Aop_bf16 = \
                        new cuCSCAffineMatrixFunction[__nv_bfloat16](
                            <__nv_bfloat16*> A_data,
                            <LongIndexType*> A_indices,
                            <LongIndexType*> A_index_pointer,
                            A_num_rows,
                            A_num_columns,
                            self.A_is_symmetric,
                            <__nv_bfloat16*> B_data,
                            <LongIndexType*> B_indices,
                            <LongIndexType*> B_index_pointer,
                            self.B_is_symmetric,
                            self.num_gpu_devices)
                ELSE:
                    raise TypeError('Package was not compiled with BF16 '
                                    'precision support for CUDA.')

        elif self.data_type_name == b'float16':

            if B_is_identity:
                IF USE_CUDA_FP16:
                    self.Aop_fp16 = new cuCSCAffineMatrixFunction[__half](
                            <__half*> A_data,
                            <LongIndexType*> A_indices,
                            <LongIndexType*> A_index_pointer,
                            A_num_rows,
                            A_num_columns,
                            self.A_is_symmetric,
                            self.num_gpu_devices)
                ELSE:
                    raise TypeError('Package was not compiled with FP16 '
                                    'precision support for CUDA.')
            else:
                IF USE_CUDA_FP16:
                    self.Aop_fp16 = new cuCSCAffineMatrixFunction[__half](
                            <__half*> A_data,
                            <LongIndexType*> A_indices,
                            <LongIndexType*> A_index_pointer,
                            A_num_rows,
                            A_num_columns,
                            self.A_is_symmetric,
                            <__half*> B_data,
                            <LongIndexType*> B_indices,
                            <LongIndexType*> B_index_pointer,
                            self.B_is_symmetric,
                            self.num_gpu_devices)
                ELSE:
                    raise TypeError('Package was not compiled with FP16 '
                                    'precision support for CUDA.')

        elif self.data_type_name == b'float32':

            if B_is_identity:
                IF USE_CUDA_FP32:
                    self.Aop_fp32 = new cuCSCAffineMatrixFunction[float](
                            <float*> A_data,
                            <LongIndexType*> A_indices,
                            <LongIndexType*> A_index_pointer,
                            A_num_rows,
                            A_num_columns,
                            self.A_is_symmetric,
                            self.num_gpu_devices)
                ELSE:
                    raise TypeError('Package was not compiled with FP32 '
                                    'precision support for CUDA.')
            else:
                IF USE_CUDA_FP32:
                    self.Aop_fp32 = new cuCSCAffineMatrixFunction[float](
                            <float*> A_data,
                            <LongIndexType*> A_indices,
                            <LongIndexType*> A_index_pointer,
                            A_num_rows,
                            A_num_columns,
                            self.A_is_symmetric,
                            <float*> B_data,
                            <LongIndexType*> B_indices,
                            <LongIndexType*> B_index_pointer,
                            self.B_is_symmetric,
                            self.num_gpu_devices)
                ELSE:
                    raise TypeError('Package was not compiled with FP32 '
                                    'precision support for CUDA.')

        elif self.data_type_name == b'float64':

            if B_is_identity:
                IF USE_CUDA_FP64:
                    self.Aop_fp64 = new cuCSCAffineMatrixFunction[double](
                            <double*> A_data,
                            <LongIndexType*> A_indices,
                            <LongIndexType*> A_index_pointer,
                            A_num_rows,
                            A_num_columns,
                            self.A_is_symmetric,
                            self.num_gpu_devices)
                ELSE:
                    raise TypeError('Package was not compiled with FP64 '
                                    'precision support for CUDA.')
            else:
                IF USE_CUDA_FP64:
                    self.Aop_fp64 = new cuCSCAffineMatrixFunction[double](
                            <double*> A_data,
                            <LongIndexType*> A_indices,
                            <LongIndexType*> A_index_pointer,
                            A_num_rows,
                            A_num_columns,
                            self.A_is_symmetric,
                            <double*> B_data,
                            <LongIndexType*> B_indices,
                            <LongIndexType*> B_index_pointer,
                            self.B_is_symmetric,
                            self.num_gpu_devices)
                ELSE:
                    raise TypeError('Package was not compiled with FP64 '
                                    'precision support for CUDA.')
