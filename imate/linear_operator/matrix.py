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
from scipy.sparse import isspmatrix


# ======
# Matrix
# ======

class Matrix(LinearOperator):
    """
    Create a linear operator object from an input matrix.

    The linear operator is a container for various matrix types with a unified
    interface, establishes a fully automatic dynamic buffer to allocate,
    deallocate, and transfer data between CPU and multiple GPU devices on
    demand, as well as performs basic matrix-vector operations with
    high performance on both CPU or GPU devices.

    An instance of this class can be used as an input matrix to any function in
    :mod:`imate` that accepts ``slq`` method.

    Parameters
    ----------

    A : numpy.ndarray or scipy.sparse, (n, n)
        The input matrix `A` can be dense or sparse (both `CSR` and `CSC`
        formats are supported). Also, the matrix data type can be either
        `32-bit`, `64-bit`, or `128-bit`. The input matrix can be stored
        either in row-ordering (`C` style) or column-ordering (`Fortran`
        style).

        .. note::

            `128-bit` data type is not supported on GPU.

    Attributes
    ----------

    A : numpy.ndarray or scipy.sparse
        Input matrix `A` from python object.

    cpu_Aop : object
        Matrix object on CPU.

    gpu_Aop : object
        Matrix object on GPU.

    gpu : bool, default=False
        If `True`, the matrix object is created for GPU devices.

    num_gpu_devices : int, default=0
        Number of GPU devices to be used. If `0`, it uses maximum number of
        GPU devices that are available.

    initialized_on_cpu : bool, default=False
        Indicates whether the matrix data is allocated in CPU.

    initialized_on_gpu : bool, default=False
        Indicates whether the matrix data is allocated in GPU.

    data_type_name : str, default=None
        The type of matrix data, and can be `float32`, `float64`, or
        `float128`.

    num_parameters : int, default=0
        Number of parameters of the linear operator. For :class:`Matrix` class,
        this parameter is always `0`.

    Methods
    -------
    initialize
    get_num_rows
    get_num_columns
    is_sparse
    get_nnz
    get_density
    get_data_type_name
    get_num_parameters
    get_linear_operator

    See Also
    --------

    imate.AffineMatrixFunction

    Notes
    -----

    **Where to use this class:**

    The instances of this class can be used just as a normal matrix in any
    function in :mod:`imate` that accepts ``slq`` method. For instance, when
    calling :func:`imate.logdet` function using ``method=slq`` argument, the
    input matrix `A` can be an instance of this class.

    **Why using this class:**

    This class is a replacement to `numpy`'s dense matrices and `scipy`'s
    sparse matrices. The purpose of creating a linear operator is three-fold:

    1. A container to handle a variety of matrices and data types with a
       unified interface.
    2. A dynamic buffer for matrix data on either CPU or multi-GPU devices
       with fully automatic allocation, deallocation, and data transfer between
       CPU and GPU devices.
    3. Performing basic linear algebraic operations on the matrix data with
       high performance on CPU or GPU devices.

    Further details of each of the points are described below.

    **1. Unified interface for matrix types:**

    The linear operator encompasses the following matrix types all in one
    interface:

      * *Dense* or *sparse* matrices with `CSR` or `CSC` sparse formats.
      * `32-bit`, `64-bit`, and `128-bit` data types.
      * Row-ordering (`C` style) and column-ordering (`Fortran` style) storage.

    **2. Unified interface for memory buffer between CPU and GPU:**

    This class creates a dynamic buffer to automatically allocate, deallocate,
    and transfer matrix data based on demand between CPU and multiple GPU
    devices. These operations are performed **in parallel for each GPU
    device**. Also, deallocation is performed by a **smart garbage collector**,
    so the user should not be concerned about cleaning the data on the device
    and the issue of memory leak. An instance of this class can be efficiently
    reused between multiple function calls. This class uses a **lazy evaluation
    strategy**, meaning that data allocation and transfer are not performed
    until a caller function requests matrix data from this class.

    **3. Unified interface for basic algebraic operations:**

    This class handles the following internal operations:

    1. Matrix vector product
       :math:`\\boldsymbol{y} = \\mathbf{A} \\boldsymbol{x}`.

    2. Transposed matrix vector product
       :math:`\\boldsymbol{y} = \\mathbf{A}^{\\intercal} \\boldsymbol{x}`.

    3. Additive matrix vector product
       :math:`\\boldsymbol{y} = \\boldsymbol{y} + \\mathbf{A} \\boldsymbol{x}`.

    4. Additive transposed matrix vector product
       :math:`\\boldsymbol{y} = \\boldsymbol{y} + \\mathbf{A}^{\\intercal}
       \\boldsymbol{x}`.

    All the above operations are handled internally when an instance of this
    class is called by other functions.

    Each of the above operations has various internal implementations depending
    on whether the matrix format is dense, sparse CSR, or sparse CSC,
    whether the memory storage is C style or Fortran style, whether the data
    type of 32-bit, 64-bit, or 128-bit, and whether the operation is performed
    on CPU or multi-GPU devices. This class unifies all these implementations
    in one interface.

    Examples
    --------

    **Create Matrix object:**

    Create a very large sparse matrix with 64-bit data type:

    .. code-block:: python

        >>> # Create a random sparse matrix with the size of ten million
        >>> from imate import toeplitz
        >>> n = 10000000
        >>> A = toeplitz(2, 1, n, gram=True, format='csr', dtype='float64')
        >>> print(A.dtype)
        dtype('float64')

        >>> print(type(A))
        scipy.sparse.csr.csr_matrix

    Create a linear operator from the matrix `A`:

    .. code-block:: python
        :emphasize-lines: 5

        >>> # Import matrix operator
        >>> from imate import Matrix

        >>> # Create a matrix operator object from matrix A
        >>> Aop = Matrix(A)

    **Operation on CPU:**

    Pass the above linear operator as input to any of the matrix functions in
    :mod:`imate` that accepts ``slq`` method, such as
    :func:`imate.logdet` as follows:

    .. code-block:: python

        >>> # Compute log-determinant of Aop
        >>> from imate import logdet
        >>> logdet(Aop, method='slq')
        13861581.003237441

    Note that the above function could also be called directly on the scipy
    object `A` rather than `Aop`:

    .. code-block:: python

        >>> # Compute log-determinant of Aop
        >>> logdet(A, method='slq')
        13861581.003237441

    However, with the above approach, the :func:`logdet` function still creates
    an object of :class:`imate.Matrix` internally. The advantage of using `Aop`
    instead of `A` is more clear when using GPU.

    **Operation on GPU:**

    Reuse the same object `Aop` as created in the previous example. However,
    this time by calling :func:`imate.logdet` function with ``gpu=True``
    argument, the data is automatically transferred from CPU to each of the
    multi-GPU devices (then, it will be available on both CPU and GPU devices).

    .. code-block:: python

        >>> # Compute log-determinant of Aop on GPU
        >>> logdet(Aop, method='slq', gpu=True)
        13861581.003237441

    **Why using imate.Matrix object:**

    An advantage of using `Aop` (instead of the matrix `A` directly) is that
    by calling the above `logdet` function (or another function) again on this
    matrix, the data of this matrix does not have to be re-allocated on the GPU
    device again. The next example highlights this point. Call a second
    function, for instance, `traceinv`, on the same object `Aop` as follows.

    .. code-block:: python

        >>> # Compute trace of inverse of Aop on GPU
        >>> from imate import traceinv
        >>> traceinv(Aop, method='slq', gpu=True)
        13861581.003237441

    In the above, no data is needed to be transferred from CPU host to GPU
    device again. However, if `A` was used instead of `Aop`, the data would
    have been transferred from CPU to GPU again for the second time. The `Aop`
    object holds the data on GPU for later use as long as this object does not
    go out of the scope of the python environment. Once the variable `Aop` goes
    out of scope, the matrix data on all the GPU devices will be cleaned
    automatically.
    """

    # ====
    # init
    # ====

    def __init__(self, A):
        """
        Initializes attributes.
        """

        # Calling base class to initialize member data
        super(Matrix, self).__init__()

        # A reference to numpy or scipt sparse matrix
        self.A = A
        self.num_parameters = 0
        self.set_data_type_name(self.A)

    # ==========
    # initialize
    # ==========

    def initialize(self, gpu, num_gpu_devices):
        """
        Initializes the object.

        Parameters
        ----------

        gpu : bool
            If `True`, the matrix array will be initialized on GPU devices.

        num_gpu_devices : int
            The number of GPU devices to use. If `0`, it uses the maximum
            number of available GPU devices.
        """

        if gpu:

            if not self.initialized_on_gpu:

                # Create a linear operator on GPU
                from .._cu_linear_operator import pycuMatrix
                self.gpu_Aop = pycuMatrix(self.A, num_gpu_devices)
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
                    from .._cu_linear_operator import pycuMatrix

                    # If allocated before, deallocate. This occurs then the
                    # number of gpu devices changes and gpu_Aop needs to be
                    # reallocated.
                    if self.gpu_Aop is not None:
                        del self.gpu_Aop

                    self.gpu_Aop = pycuMatrix(self.A, num_gpu_devices)
                    self.initialized_on_gpu = True

        else:

            if not self.initialized_on_cpu:

                # Create a linear operator on CPU
                from .._c_linear_operator import pycMatrix
                self.cpu_Aop = pycMatrix(self.A)
                self.initialized_on_cpu = True

        self.gpu = gpu
        self.num_gpu_devices = num_gpu_devices

    # ============
    # get num rows
    # ============

    def get_num_rows(self):
        """
        Returns the number of rows of the matrix.

        Returns
        -------

        num_rows : int
            Number of rows of the matrix.
        """

        return self.A.shape[0]

    # ===============
    # get num columns
    # ===============

    def get_num_columns(self):
        """
        Returns the number of columns of the matrix.

        Returns
        -------

        num_columns : int
            Number of columns of the matrix.
        """

        return self.A.shape[1]

    # =========
    # is sparse
    # =========

    def is_sparse(self):
        """
        Determines whether the matrix is dense or sparse.

        Returns
        -------

        sparse : bool
            If `True`, the matrix is sparse. Otherwise, the matrix is dense.
        """

        return isspmatrix(self.A)

    # =======
    # get nnz
    # =======

    def get_nnz(self):
        """
        Returns the number of non-zero elements of the matrix.

        Returns
        -------

        nnz : int
            Number of nonzero elements of the matrix.
        """

        if self.is_sparse():
            return self.A.nnz
        else:
            return self.get_num_rows() * self.get_num_columns()

    # ===========
    # get density
    # ===========

    def get_density(self):
        """
        Returns the density of non-zero elements of the matrix.

        Returns
        -------

        density : float
            The density of the nonzero elements of the sparse matrix. If the
            matrix is dense, the density is `1`.
        """

        if self.is_sparse():
            return self.get_nnz() / \
                (self.get_num_rows() * self.get_num_columns())
        else:
            return 1.0
