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
from scipy.sparse import issparse, isspmatrix_csr, isspmatrix_csc, csr_matrix

# Cython
from .py_cu_linear_operator cimport pycuLinearOperator
from .cu_dense_affine_matrix_function cimport cuDenseAffineMatrixFunction
from .cu_csr_affine_matrix_function cimport cuCSRAffineMatrixFunction
from .cu_csc_affine_matrix_function cimport cuCSCAffineMatrixFunction
from .._definitions.types cimport IndexType, LongIndexType, FlagType, \
        MemoryViewLongIndexType


# ===========================
# pycu Affine Matrix Function
# ===========================

cdef class pycuAffineMatrixFunction(pycuLinearOperator):
    """
    Defines a linear operator that is an affine function of a single parameter.
    Given two matrices :math:`\\mathbf{A}` and :math:`\\mathf{B}`, the linear
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

    def __cinit__(self, A, B=None, num_gpu_devices=0):
        """
        Sets matrices A and B.
        """

        # Number of gpu devices to use. This might be different (less) than the
        # number of gpu devices that are avalable. If set to 0, all available
        # devices will be used.
        self.num_gpu_devices = num_gpu_devices

        # Check A
        if A is None:
            raise ValueError('A cannot be None.')

        if A.ndim != 2:
            raise ValueError('Input matrix should be a 2-dimensional array.')

        # Data type
        if A.dtype == b'float32':
            self.data_type_name = b'float32'
        elif A.dtype == b'float64':
            self.data_type_name = b'float64'
        else:
            raise TypeError('When gpu is enabled, data type should be ' +
                            '"float32" or "float64".')

        # Check if B is noe to be considered as identity matrix
        if B is None:

            # B is assumed to be identity
            B_is_identity = True

        else:

            # B is neither zero nor identity
            B_is_identity = False

            # Check similar types of A and B
            if not (type(A) == type(B)):
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
            if isspmatrix_csr(A):

                # Check sorted indices
                if not A.has_sorted_indices:
                    A.sort_indices()

                if (not B_is_identity) and (not B.has_sorted_indices):
                    B.sort_indices()

                # CSR matrix
                if self.data_type_name == b'float32':
                    self.set_csr_matrix_float(A, B, B_is_identity)

                elif self.data_type_name == b'float64':
                    self.set_csr_matrix_double(A, B, B_is_identity)

            elif isspmatrix_csc(A):

                # Check sorted indices
                if not A.has_sorted_indices:
                    A.sort_indices()

                if (not B_is_identity) and (not B.has_sorted_indices):
                    B.sort_indices()

                # CSC matrix
                if self.data_type_name == b'float32':
                    self.set_csc_matrix_float(A, B, B_is_identity)

                elif self.data_type_name == b'float64':
                    self.set_csc_matrix_double(A, B, B_is_identity)

            else:

                # If A is neither CSR or CSC, convert A to CSR
                self.A_csr = csr_matrix(A)

                if not B_is_identity:
                    self.B_csr = csr_matrix(B)
                else:
                    self.B_csr = B

                # Check sorted indices
                if not self.A_csr.has_sorted_indices:
                    self.A_csr.sort_indices()

                if (not B_is_identity) and (not self.B_csr.has_sorted_indices):
                    self.B_csr.sort_indices()

                # CSR matrix
                if self.data_type_name == b'float32':
                    self.set_csr_matrix_float(self.A_csr, self.B_csr,
                                              B_is_identity)

                elif self.data_type_name == b'float64':
                    self.set_csr_matrix_double(self.A_csr, self.B_csr,
                                               B_is_identity)

        else:

            # Set a dense matrix
            if self.data_type_name == b'float32':
                self.set_dense_matrix_float(A, B, B_is_identity)

            elif self.data_type_name == b'float64':
                self.set_dense_matrix_double(A, B, B_is_identity)

    # ======================
    # set dense matrix float
    # ======================

    def set_dense_matrix_float(self, A, B, B_is_identity):
        """
        Sets matrix A.

        :param A: A 2-dimensional matrix.
        :type A: numpy.ndarray, or any scipy.sparse array
        """

        # Matrix size
        cdef LongIndexType A_num_rows = A.shape[0]
        cdef LongIndexType A_num_columns = A.shape[1]

        # Contiguity
        cdef FlagType A_is_row_major
        cdef FlagType B_is_row_major = 0

        if A.flags['C_CONTIGUOUS']:
            A_is_row_major = 1
        elif A.flags['F_CONTIGUOUS']:
            A_is_row_major = 0
        else:
            raise TypeError('Matrix A should be either C or F contiguous.')

        if not B_is_identity:
            if B.flags['C_CONTIGUOUS']:
                B_is_row_major = 1
            elif B.flags['F_CONTIGUOUS']:
                B_is_row_major = 0
            else:
                raise TypeError('Matrix B should be either C or F contiguous.')

        # Declare memoryviews to get data pointer
        cdef float[:, ::1] A_data_mv_c
        cdef float[::1, :] A_data_mv_f
        cdef float[:, ::1] B_data_mv_c = None
        cdef float[::1, :] B_data_mv_f = None

        # Declare pointer of A.data and B.data
        cdef float* A_data
        cdef float* B_data = NULL

        # Get pointer to data of A depending on row or column major
        if A_is_row_major:

            # Memoryview of A for row major matrix
            A_data_mv_c = A

            # Pointer of the data of A
            A_data = &A_data_mv_c[0, 0]

        else:

            # Memoryview of A for column major matrix
            A_data_mv_f = A

            # Pointer of the data of A
            A_data = &A_data_mv_f[0, 0]

        # Get pointer to data of B depending on row or column major
        if not B_is_identity:
            if B_is_row_major:

                # Memoryview of B for row major matrix
                B_data_mv_c = B

                # Pointer of the data of B
                B_data = &B_data_mv_c[0, 0]

            else:

                # Memoryview of B for column major matrix
                B_data_mv_f = B

                # Pointer of the data of B
                B_data = &B_data_mv_f[0, 0]

        # Create a linear operator object
        if B_is_identity:
            self.Aop_float = new cuDenseAffineMatrixFunction[float](
                    A_data,
                    A_is_row_major,
                    A_num_rows,
                    A_num_columns,
                    self.num_gpu_devices)
        else:
            self.Aop_float = new cuDenseAffineMatrixFunction[float](
                    A_data,
                    A_is_row_major,
                    A_num_rows,
                    A_num_columns,
                    B_data,
                    B_is_row_major,
                    self.num_gpu_devices)

    # =======================
    # set dense matrix double
    # =======================

    def set_dense_matrix_double(self, A, B, B_is_identity):
        """
        Sets matrix A.

        :param A: A 2-dimensional matrix.
        :type A: numpy.ndarray, or any scipy.sparse array
        """

        # Matrix size
        cdef LongIndexType A_num_rows = A.shape[0]
        cdef LongIndexType A_num_columns = A.shape[1]

        # Contiguity
        cdef FlagType A_is_row_major
        cdef FlagType B_is_row_major = 0

        if A.flags['C_CONTIGUOUS']:
            A_is_row_major = 1
        elif A.flags['F_CONTIGUOUS']:
            A_is_row_major = 0
        else:
            raise TypeError('Matrix A should be either C or F contiguous.')

        if not B_is_identity:
            if B.flags['C_CONTIGUOUS']:
                B_is_row_major = 1
            elif B.flags['F_CONTIGUOUS']:
                B_is_row_major = 0
            else:
                raise TypeError('Matrix B should be either C or F contiguous.')

        # Declare memoryviews to get data pointer
        cdef double[:, ::1] A_data_mv_c
        cdef double[::1, :] A_data_mv_f
        cdef double[:, ::1] B_data_mv_c = None
        cdef double[::1, :] B_data_mv_f = None

        # Declare pointer to A.data and B.data
        cdef double* A_data
        cdef double* B_data = NULL

        # Get pointer to data of A depending on row or column major
        if A_is_row_major:

            # Memoryview of A for row major matrix
            A_data_mv_c = A

            # Pointer of the data of A
            A_data = &A_data_mv_c[0, 0]

        else:

            # Memoryview of A for column major matrix
            A_data_mv_f = A

            # Pointer of the data of A
            A_data = &A_data_mv_f[0, 0]

        # Get pointer to data of B depending on row or column major
        if not B_is_identity:
            if B_is_row_major:

                # Memoryview of B for row major matrix
                B_data_mv_c = B

                # Pointer of the data of B
                B_data = &B_data_mv_c[0, 0]

            else:

                # Memoryview of B for column major matrix
                B_data_mv_f = B

                # Pointer of the data of B
                B_data = &B_data_mv_f[0, 0]

        # Create a linear operator object
        if B_is_identity:
            self.Aop_double = new cuDenseAffineMatrixFunction[double](
                    A_data,
                    A_is_row_major,
                    A_num_rows,
                    A_num_columns,
                    self.num_gpu_devices)
        else:
            self.Aop_double = new cuDenseAffineMatrixFunction[double](
                    A_data,
                    A_is_row_major,
                    A_num_rows,
                    A_num_columns,
                    B_data,
                    B_is_row_major,
                    self.num_gpu_devices)

    # ====================
    # set csr matrix float
    # ====================

    def set_csr_matrix_float(self, A, B, B_is_identity):
        """
        """

        # Matrix size
        cdef LongIndexType A_num_rows = A.shape[0]
        cdef LongIndexType A_num_columns = A.shape[1]

        # If the input type is the same as LongIndexType, no copy is performed.
        self.A_indices_copy = \
            A.indices.astype(self.long_index_type_name, copy=False)
        self.A_index_pointer_copy = \
            A.indptr.astype(self.long_index_type_name, copy=False)

        # Declare memoryviews to get pointers
        cdef float[:] A_data_mv = A.data
        cdef MemoryViewLongIndexType A_indices_mv = self.A_indices_copy
        cdef MemoryViewLongIndexType A_index_pointer_mv = \
            self.A_index_pointer_copy
        cdef float[:] B_data_mv = None
        cdef MemoryViewLongIndexType B_indices_mv = None
        cdef MemoryViewLongIndexType B_index_pointer_mv = None

        if not B_is_identity:

            # If input type is the same as LongIndexType, no copy is performed.
            self.B_indices_copy = \
                B.indices.astype(self.long_index_type_name, copy=False)
            self.B_index_pointer_copy = \
                B.indptr.astype(self.long_index_type_name, copy=False)

            B_data_mv = B.data
            B_indices_mv = self.B_indices_copy
            B_index_pointer_mv = self.B_index_pointer_copy

        # Declare pointers
        cdef float* A_data = &A_data_mv[0]
        cdef LongIndexType* A_indices = &A_indices_mv[0]
        cdef LongIndexType* A_index_pointer = &A_index_pointer_mv[0]
        cdef float* B_data = NULL
        cdef LongIndexType* B_indices = NULL
        cdef LongIndexType* B_index_pointer = NULL

        if not B_is_identity:
            B_data = &B_data_mv[0]
            B_indices = &B_indices_mv[0]
            B_index_pointer = &B_index_pointer_mv[0]

        # Create a linear operator object
        if B_is_identity:
            self.Aop_float = new cuCSRAffineMatrixFunction[float](
                    A_data,
                    A_indices,
                    A_index_pointer,
                    A_num_rows,
                    A_num_columns,
                    self.num_gpu_devices)
        else:
            self.Aop_float = new cuCSRAffineMatrixFunction[float](
                    A_data,
                    A_indices,
                    A_index_pointer,
                    A_num_rows,
                    A_num_columns,
                    B_data,
                    B_indices,
                    B_index_pointer,
                    self.num_gpu_devices)

    # =====================
    # set csr matrix double
    # =====================

    def set_csr_matrix_double(self, A, B, B_is_identity):
        """
        """

        # Matrix size
        cdef LongIndexType A_num_rows = A.shape[0]
        cdef LongIndexType A_num_columns = A.shape[1]

        # If the input type is the same as LongIndexType, no copy is performed.
        self.A_indices_copy = \
            A.indices.astype(self.long_index_type_name, copy=False)
        self.A_index_pointer_copy = \
            A.indptr.astype(self.long_index_type_name, copy=False)

        # Declare memoryviews to get pointers
        cdef double[:] A_data_mv = A.data
        cdef MemoryViewLongIndexType A_indices_mv = self.A_indices_copy
        cdef MemoryViewLongIndexType A_index_pointer_mv = \
            self.A_index_pointer_copy
        cdef double[:] B_data_mv = None
        cdef MemoryViewLongIndexType B_indices_mv = None
        cdef MemoryViewLongIndexType B_index_pointer_mv = None

        if not B_is_identity:

            # If input type is the same as LongIndexType, no copy is performed.
            self.B_indices_copy = \
                B.indices.astype(self.long_index_type_name, copy=False)
            self.B_index_pointer_copy = \
                B.indptr.astype(self.long_index_type_name, copy=False)

            B_data_mv = B.data
            B_indices_mv = self.B_indices_copy
            B_index_pointer_mv = self.B_index_pointer_copy

        # Declare pointers
        cdef double* A_data = &A_data_mv[0]
        cdef LongIndexType* A_indices = &A_indices_mv[0]
        cdef LongIndexType* A_index_pointer = &A_index_pointer_mv[0]
        cdef double* B_data = NULL
        cdef LongIndexType* B_indices = NULL
        cdef LongIndexType* B_index_pointer = NULL

        if not B_is_identity:
            B_data = &B_data_mv[0]
            B_indices = &B_indices_mv[0]
            B_index_pointer = &B_index_pointer_mv[0]

        # Create a linear operator object
        if B_is_identity:
            self.Aop_double = new cuCSRAffineMatrixFunction[double](
                    A_data,
                    A_indices,
                    A_index_pointer,
                    A_num_rows,
                    A_num_columns,
                    self.num_gpu_devices)
        else:
            self.Aop_double = new cuCSRAffineMatrixFunction[double](
                    A_data,
                    A_indices,
                    A_index_pointer,
                    A_num_rows,
                    A_num_columns,
                    B_data,
                    B_indices,
                    B_index_pointer,
                    self.num_gpu_devices)

    # ====================
    # set csc matrix float
    # ====================

    def set_csc_matrix_float(self, A, B, B_is_identity):
        """
        """

        # Matrix size
        cdef LongIndexType A_num_rows = A.shape[0]
        cdef LongIndexType A_num_columns = A.shape[1]

        # If the input type is the same as LongIndexType, no copy is performed.
        self.A_indices_copy = \
            A.indices.astype(self.long_index_type_name, copy=False)
        self.A_index_pointer_copy = \
            A.indptr.astype(self.long_index_type_name, copy=False)

        # Declare memoryviews to get pointers
        cdef float[:] A_data_mv = A.data
        cdef MemoryViewLongIndexType A_indices_mv = self.A_indices_copy
        cdef MemoryViewLongIndexType A_index_pointer_mv = \
            self.A_index_pointer_copy
        cdef float[:] B_data_mv = None
        cdef MemoryViewLongIndexType B_indices_mv = None
        cdef MemoryViewLongIndexType B_index_pointer_mv = None

        if not B_is_identity:

            # If input type is the same as LongIndexType, no copy is performed.
            self.B_indices_copy = \
                B.indices.astype(self.long_index_type_name, copy=False)
            self.B_index_pointer_copy = \
                B.indptr.astype(self.long_index_type_name, copy=False)

            B_data_mv = B.data
            B_indices_mv = self.B_indices_copy
            B_index_pointer_mv = self.B_index_pointer_copy

        # Declare pointers
        cdef float* A_data = &A_data_mv[0]
        cdef LongIndexType* A_indices = &A_indices_mv[0]
        cdef LongIndexType* A_index_pointer = &A_index_pointer_mv[0]
        cdef float* B_data = NULL
        cdef LongIndexType* B_indices = NULL
        cdef LongIndexType* B_index_pointer = NULL

        if not B_is_identity:
            B_data = &B_data_mv[0]
            B_indices = &B_indices_mv[0]
            B_index_pointer = &B_index_pointer_mv[0]

        # Create a linear operator object
        if B_is_identity:
            self.Aop_float = new cuCSCAffineMatrixFunction[float](
                    A_data,
                    A_indices,
                    A_index_pointer,
                    A_num_rows,
                    A_num_columns,
                    self.num_gpu_devices)
        else:
            self.Aop_float = new cuCSCAffineMatrixFunction[float](
                    A_data,
                    A_indices,
                    A_index_pointer,
                    A_num_rows,
                    A_num_columns,
                    B_data,
                    B_indices,
                    B_index_pointer,
                    self.num_gpu_devices)

    # =====================
    # set csc matrix double
    # =====================

    def set_csc_matrix_double(self, A, B, B_is_identity):
        """
        """

        # Matrix size
        cdef LongIndexType A_num_rows = A.shape[0]
        cdef LongIndexType A_num_columns = A.shape[1]

        # If the input type is the same as LongIndexType, no copy is performed.
        self.A_indices_copy = \
            A.indices.astype(self.long_index_type_name, copy=False)
        self.A_index_pointer_copy = \
            A.indptr.astype(self.long_index_type_name, copy=False)

        # Declare memoryviews to get pointers
        cdef double[:] A_data_mv = A.data
        cdef MemoryViewLongIndexType A_indices_mv = self.A_indices_copy
        cdef MemoryViewLongIndexType A_index_pointer_mv = \
            self.A_index_pointer_copy
        cdef double[:] B_data_mv = None
        cdef MemoryViewLongIndexType B_indices_mv = None
        cdef MemoryViewLongIndexType B_index_pointer_mv = None

        if not B_is_identity:

            # If input type is the same as LongIndexType, no copy is performed.
            self.B_indices_copy = \
                B.indices.astype(self.long_index_type_name, copy=False)
            self.B_index_pointer_copy = \
                B.indptr.astype(self.long_index_type_name, copy=False)

            B_data_mv = B.data
            B_indices_mv = self.B_indices_copy
            B_index_pointer_mv = self.B_index_pointer_copy

        # Declare pointers
        cdef double* A_data = &A_data_mv[0]
        cdef LongIndexType* A_indices = &A_indices_mv[0]
        cdef LongIndexType* A_index_pointer = &A_index_pointer_mv[0]
        cdef double* B_data = NULL
        cdef LongIndexType* B_indices = NULL
        cdef LongIndexType* B_index_pointer = NULL

        if not B_is_identity:
            B_data = &B_data_mv[0]
            B_indices = &B_indices_mv[0]
            B_index_pointer = &B_index_pointer_mv[0]

        # Create a linear operator object
        if B_is_identity:
            self.Aop_double = new cuCSCAffineMatrixFunction[double](
                    A_data,
                    A_indices,
                    A_index_pointer,
                    A_num_rows,
                    A_num_columns,
                    self.num_gpu_devices)
        else:
            self.Aop_double = new cuCSCAffineMatrixFunction[double](
                    A_data,
                    A_indices,
                    A_index_pointer,
                    A_num_rows,
                    A_num_columns,
                    B_data,
                    B_indices,
                    B_index_pointer,
                    self.num_gpu_devices)
