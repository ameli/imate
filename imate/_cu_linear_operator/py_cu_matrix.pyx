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
from .cu_matrix cimport cuMatrix
from .cu_dense_matrix cimport cuDenseMatrix
from .cu_csr_matrix cimport cuCSRMatrix
from .cu_csc_matrix cimport cuCSCMatrix
from .._definitions.types cimport IndexType, LongIndexType, FlagType, \
        MemoryViewLongIndexType


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

    def __cinit__(self, A, num_gpu_devices=0):
        """
        Sets the matrix A.
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

        # Determine A is sparse or dense
        if issparse(A):

            # Matrix type codes: 'r' for CSR, and 'c' for CSC
            if isspmatrix_csr(A):

                # Check sorted indices
                if not A.has_sorted_indices:
                    A.sort_indices()

                # CSR matrix
                if self.data_type_name == b'float32':
                    self.set_csr_matrix_float(A)

                elif self.data_type_name == b'float64':
                    self.set_csr_matrix_double(A)

            elif isspmatrix_csc(A):

                # Check sorted indices
                if not A.has_sorted_indices:
                    A.sort_indices()

                # CSC matrix
                if self.data_type_name == b'float32':
                    self.set_csc_matrix_float(A)

                elif self.data_type_name == b'float64':
                    self.set_csc_matrix_double(A)

            else:

                # If A is neither CSR or CSC, convert A to CSR
                self.A_csr = csr_matrix(A)

                # Check sorted indices
                if not self.A_csr.has_sorted_indices:
                    self.A_csr.sort_indices()

                # CSR matrix
                if self.data_type_name == b'float32':
                    self.set_csr_matrix_float(self.A_csr)

                elif self.data_type_name == b'float64':
                    self.set_csr_matrix_double(self.A_csr)

        else:

            # Set a dense matrix
            if self.data_type_name == b'float32':
                self.set_dense_matrix_float(A)

            elif self.data_type_name == b'float64':
                self.set_dense_matrix_double(A)

    # ======================
    # set dense matrix float
    # ======================

    def set_dense_matrix_float(self, A):
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
        if A.flags['C_CONTIGUOUS']:
            A_is_row_major = 1
        elif A.flags['F_CONTIGUOUS']:
            A_is_row_major = 0
        else:
            raise TypeError('Matrix A should be either C or F contiguous.')

        # Declare memoryviews to get data pointer
        cdef float[:, ::1] A_data_float_mv_c
        cdef float[::1, :] A_data_float_mv_f

        # Declare pointer of A.data
        cdef float* A_data_float

        # Get pointer to data of A depending on row or column major
        if A_is_row_major:

            # Memoryview of A for row major matrix
            A_data_float_mv_c = A

            # Pointer of the data of A
            A_data_float = &A_data_float_mv_c[0, 0]

        else:

            # Memoryview of A for column major matrix
            A_data_float_mv_f = A

            # Pointer of the data of A
            A_data_float = &A_data_float_mv_f[0, 0]

        # Create a linear operator object
        self.Aop_float = new cuDenseMatrix[float](
                A_data_float,
                A_num_rows,
                A_num_columns,
                A_is_row_major,
                self.num_gpu_devices)

    # =======================
    # set dense matrix double
    # =======================

    def set_dense_matrix_double(self, A):
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
        if A.flags['C_CONTIGUOUS']:
            A_is_row_major = 1
        elif A.flags['F_CONTIGUOUS']:
            A_is_row_major = 0
        else:
            raise TypeError('Matrix A should be either C or F contiguous.')

        # Declare memoryviews to get data pointer
        cdef double[:, ::1] A_data_double_mv_c
        cdef double[::1, :] A_data_double_mv_f

        # Declare pointer to A.data
        cdef double* A_data_double

        # Get pointer to data of A depending on row or column major
        if A_is_row_major:

            # Memoryview of A for row major matrix
            A_data_double_mv_c = A

            # Pointer of the data of A
            A_data_double = &A_data_double_mv_c[0, 0]

        else:

            # Memoryview of A for column major matrix
            A_data_double_mv_f = A

            # Pointer of the data of A
            A_data_double = &A_data_double_mv_f[0, 0]

        # Create a linear operator object
        self.Aop_double = new cuDenseMatrix[double](
                A_data_double,
                A_num_rows,
                A_num_columns,
                A_is_row_major,
                self.num_gpu_devices)

    # ====================
    # set csr matrix float
    # ====================

    def set_csr_matrix_float(self, A):
        """
        """

        # Matrix size
        cdef LongIndexType A_num_rows = A.shape[0]
        cdef LongIndexType A_num_columns = A.shape[1]

        # Declare memoryviews to get pointer of A.data
        cdef float[:] A_data_float_mv

        # Declare pointer for A.data
        cdef float* A_data_float

        # If the input type is the same as LongIndexType, no copy is performed.
        self.A_indices_copy = \
            A.indices.astype(self.long_index_type_name, copy=False)
        self.A_index_pointer_copy = \
            A.indptr.astype(self.long_index_type_name, copy=False)

        # Declare memoryviews to get pointer of A.indices and A.indptr
        cdef MemoryViewLongIndexType A_indices_mv = self.A_indices_copy
        cdef MemoryViewLongIndexType A_index_pointer_mv = \
            self.A_index_pointer_copy

        # Declare pointers to A.indices ans A.indptr
        cdef LongIndexType* A_indices = &A_indices_mv[0]
        cdef LongIndexType* A_index_pointer = &A_index_pointer_mv[0]

        # Memoryview of A data
        A_data_float_mv = A.data

        # Get pointers
        A_data_float = &A_data_float_mv[0]

        # Create a linear operator object
        self.Aop_float = new cuCSRMatrix[float](
                A_data_float,
                A_indices,
                A_index_pointer,
                A_num_rows,
                A_num_columns,
                self.num_gpu_devices)

    # =====================
    # set csr matrix double
    # =====================

    def set_csr_matrix_double(self, A):
        """
        """

        # Matrix size
        cdef LongIndexType A_num_rows = A.shape[0]
        cdef LongIndexType A_num_columns = A.shape[1]

        # Declare memoryviews to get pointer of A.data
        cdef double[:] A_data_double_mv

        # Declare pointer for A.data
        cdef double* A_data_double

        # If the input type is the same as LongIndexType, no copy is performed.
        self.A_indices_copy = \
            A.indices.astype(self.long_index_type_name, copy=False)
        self.A_index_pointer_copy = \
            A.indptr.astype(self.long_index_type_name, copy=False)

        # Declare memoryviews to get pointer of A.indices and A.indptr
        cdef MemoryViewLongIndexType A_indices_mv = self.A_indices_copy
        cdef MemoryViewLongIndexType A_index_pointer_mv = \
            self.A_index_pointer_copy

        # Declare pointers to A.indices ans A.indptr
        cdef LongIndexType* A_indices = &A_indices_mv[0]
        cdef LongIndexType* A_index_pointer = &A_index_pointer_mv[0]

        # Memoryview of A data
        A_data_double_mv = A.data

        # Get pointers
        A_data_double = &A_data_double_mv[0]

        # Create a linear operator object
        self.Aop_double = new cuCSRMatrix[double](
                A_data_double,
                A_indices,
                A_index_pointer,
                A_num_rows,
                A_num_columns,
                self.num_gpu_devices)

    # ====================
    # set csc matrix float
    # ====================

    def set_csc_matrix_float(self, A):
        """
        """

        # Matrix size
        cdef LongIndexType A_num_rows = A.shape[0]
        cdef LongIndexType A_num_columns = A.shape[1]

        # Declare memoryviews to get pointer of A.data
        cdef float[:] A_data_float_mv

        # Declare pointer for A.data
        cdef float* A_data_float

        # If the input type is the same as LongIndexType, no copy is performed.
        self.A_indices_copy = \
            A.indices.astype(self.long_index_type_name, copy=False)
        self.A_index_pointer_copy = \
            A.indptr.astype(self.long_index_type_name, copy=False)

        # Declare memoryviews to get pointer of A.indices and A.indptr
        cdef MemoryViewLongIndexType A_indices_mv = self.A_indices_copy
        cdef MemoryViewLongIndexType A_index_pointer_mv = \
            self.A_index_pointer_copy

        # Declare pointers to A.indices ans A.indptr
        cdef LongIndexType* A_indices = &A_indices_mv[0]
        cdef LongIndexType* A_index_pointer = &A_index_pointer_mv[0]

        # Memoryview of A data
        A_data_float_mv = A.data

        # Get pointers
        A_data_float = &A_data_float_mv[0]

        # Create a linear operator object
        self.Aop_float = new cuCSCMatrix[float](
                A_data_float,
                A_indices,
                A_index_pointer,
                A_num_rows,
                A_num_columns,
                self.num_gpu_devices)

    # =====================
    # set csc matrix double
    # =====================

    def set_csc_matrix_double(self, A):
        """
        """

        # Matrix size
        cdef LongIndexType A_num_rows = A.shape[0]
        cdef LongIndexType A_num_columns = A.shape[1]

        # Declare memoryviews to get pointer of A.data
        cdef double[:] A_data_double_mv

        # Declare pointer for A.data
        cdef double* A_data_double

        # If the input type is the same as LongIndexType, no copy is performed.
        self.A_indices_copy = \
            A.indices.astype(self.long_index_type_name, copy=False)
        self.A_index_pointer_copy = \
            A.indptr.astype(self.long_index_type_name, copy=False)

        # Declare memoryviews to get pointer of A.indices and A.indptr
        cdef MemoryViewLongIndexType A_indices_mv = self.A_indices_copy
        cdef MemoryViewLongIndexType A_index_pointer_mv = \
            self.A_index_pointer_copy

        # Declare pointers to A.indices ans A.indptr
        cdef LongIndexType* A_indices = &A_indices_mv[0]
        cdef LongIndexType* A_index_pointer = &A_index_pointer_mv[0]

        # Memoryview of A data
        A_data_double_mv = A.data

        # Get pointers
        A_data_double = &A_data_double_mv[0]

        # Create a linear operator object
        self.Aop_double = new cuCSCMatrix[double](
                A_data_double,
                A_indices,
                A_index_pointer,
                A_num_rows,
                A_num_columns,
                self.num_gpu_devices)
