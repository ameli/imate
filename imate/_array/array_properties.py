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
import scipy.sparse

__all__ = ['get_device', 'get_shape', 'get_size', 'get_ndim', 'is_row_major',
           'get_data_type_name']


# ==========
# get device
# ==========

def get_device(array):
    """
    Returns the device where the array resides.

    Parameters
    ----------

    array : numpy.ndarray or torch.Tensor or tensorflow.Tensor or JAX.Tensor
        An array

    Returns
    -------

    device : str {'cpu', 'gpu'}
        Either 'cpu' or 'gpu'.

    See Also
    --------

    get_array_buffer
    """

    if isinstance(array, numpy.ndarray) or scipy.sparse.issparse(array):
        device = 'cpu'

    elif ((type(array).__module__ == 'torch') and
          ('Tensor' in type(array).__name__)):
        device = array.device.type

    elif (('tensorflow' in type(array).__module__) and
          ('Tensor' in type(array).__name__)):

        if 'GPU' in array.device:
            device = 'gpu'
        elif 'GCPU' in array.device:
            device = 'cpu'

    elif (('jaxlib' in type(array).__module__) and
          ('ArrayImpl' in type(array).__name__)):

        # Ensure any asynchronous operations are complete and on CPU
        array.block_until_ready()
        device = 'cpu'

    else:
        raise RuntimeError('Matrix type is not supported.')

    return device


# =========
# get shape
# =========

def get_shape(array):
    """
    Returns the shape of an array.

    Parameters
    ----------

    array : numpy.ndarray or torch.Tensor or tensorflow.Tensor or JAX.Tensor
        An array

    Returns
    -------

    shape : tuple
        A tuple of shape of array

    See Also
    --------

    get_size
    get_ndim
    """

    if isinstance(array, numpy.ndarray) or scipy.sparse.issparse(array):
        shape = array.shape

    elif ((type(array).__module__ == 'torch') and
          ('Tensor' in type(array).__name__)):
        shape = tuple(array.shape)

    elif (('tensorflow' in type(array).__module__) and
          ('Tensor' in type(array).__name__)):
        shape = tuple(array.shape)

    elif (('jaxlib' in type(array).__module__) and
          ('ArrayImpl' in type(array).__name__)):
        shape = array.shape

    else:
        raise RuntimeError('Matrix type is not supported.')

    return shape


# ========
# get size
# ========

def get_size(array):
    """
    Returns the size of an array.

    Parameters
    ----------

    array : numpy.ndarray or torch.Tensor or tensorflow.Tensor or JAX.Tensor
        An array

    Returns
    -------

    size : int
        Size of array, which is the product of shape elements.

    See Also
    --------

    get_shape
    get_ndim

    Notes
    -----

    For sparse matrices, the size is the number of non-zeros, not the product
    of the number of columns and rows.
    """

    if isinstance(array, numpy.ndarray) or scipy.sparse.issparse(array):
        size = array.size

    elif ((type(array).__module__ == 'torch') and
          ('Tensor' in type(array).__name__)):
        size = array.size().numel()

    elif (('tensorflow' in type(array).__module__) and
          ('Tensor' in type(array).__name__)):
        shape = tuple(array.shape)
        size = 1
        for dim in shape:
            size *= dim

    elif (('jaxlib' in type(array).__module__) and
          ('ArrayImpl' in type(array).__name__)):
        size = array.size
    else:
        raise RuntimeError('Matrix type is not supported.')

    return size


# ========
# get ndim
# ========

def get_ndim(array):
    """
    Returns the size of an array.

    Parameters
    ----------

    array : numpy.ndarray or torch.Tensor or tensorflow.Tensor or JAX.Tensor
        An array

    Returns
    -------

    size : int
        Size of array, which is the product of shape elements.

    See Also
    --------

    get_shape
    get_ndim
    """

    if isinstance(array, numpy.ndarray) or scipy.sparse.issparse(array):
        ndim = array.ndim

    elif ((type(array).__module__ == 'torch') and
          ('Tensor' in type(array).__name__)):
        ndim = array.ndim

    elif (('tensorflow' in type(array).__module__) and
          ('Tensor' in type(array).__name__)):
        ndim = array.ndim

    elif (('jaxlib' in type(array).__module__) and
          ('ArrayImpl' in type(array).__name__)):
        ndim = array.ndim
    else:
        raise RuntimeError('Matrix type is not supported.')
        ndim = array.ndim

    return ndim


# ============
# is row major
# ============

def is_row_major(array):
    """
    Checks if an array (or tensor) is row-major or column-major.

    Parameters
    ----------

    array : numpy.ndarray or torch.Tensor or tensorflow.Tensor or JAX.Tensor
        An array

    Returns
    -------

    row_major : bool
        If `True`, the array is row-major, if `False`, the array is
        column-major.

    See Also
    --------

    get_shape
    get_array_buffer
    """

    if isinstance(array, numpy.ndarray):

        if array.flags['C_CONTIGUOUS']:
            row_major = True
        elif array.flags['F_CONTIGUOUS']:
            row_major = False
        else:
            raise TypeError('Matrix should be either C-contiguous (row-major) '
                            'or F-contiguous (column-major).')

    elif scipy.sparse.issparse(array):

        if array.data.flags['C_CONTIGUOUS']:
            row_major = True
        elif array.data.flags['F_CONTIGUOUS']:
            row_major = False
        else:
            raise TypeError('Matrix should be either C-contiguous (row-major) '
                            'or F-contiguous (column-major).')

    elif ((type(array).__module__ == 'torch') and
          ('Tensor' in type(array).__name__)):

        if array.is_contiguous():
            row_major = True
        elif array.t.is_contiguous():
            row_major = False
        else:
            raise TypeError('Matrix should be either C-contiguous (row-major) '
                            'or F-contiguous (column-major).')

    elif (('tensorflow' in type(array).__module__) and
          ('Tensor' in type(array).__name__)):

        # Convert to numpy. If on GPU, this moves data from GPU to CPU. If
        # on CPU, the numpy is just a wrap without copying new data.
        array_tf = array.numpy()
        if array_tf.flags['C_CONTIGUOUS']:
            row_major = True
        elif array_tf.flags['F_CONTIGUOUS']:
            row_major = False
        else:
            raise TypeError('Matrix should be either C-contiguous (row-major) '
                            'or F-contiguous (column-major).')

    elif (('jaxlib' in type(array).__module__) and
          ('ArrayImpl' in type(array).__name__)):

        # Ensure any asynchronous operations are complete and on CPU
        array.block_until_ready()

        # From JAX tensor to numpy array
        array_jax = numpy.asarray(array)
        if array_jax.flags['C_CONTIGUOUS']:
            row_major = True
        elif array_jax.flags['F_CONTIGUOUS']:
            row_major = False
        else:
            raise TypeError('Matrix should be either C-contiguous (row-major) '
                            'or F-contiguous (column-major).')

    else:
        raise RuntimeError('Matrix type is not supported.')

    return row_major


# ==================
# get data type name
# ==================

def get_data_type_name(array):
    """
    Returns the data type of an array.

    Parameters
    ----------

    array : numpy.ndarray or torch.Tensor or tensorflow.Tensor or JAX.Tensor
        An array

    Returns
    -------

    data_type_name : str
        The string values can be as follows:
        * ``'float8_e5m2'``: corresponding to 8-bit precision consisting of
          5-bit for exponent and 2-bit for mantissa.
        * ``'float8_e4m3'``: corresponding to 8-bit precision consisting of
          4-bit for exponent and 3-bit for mantissa.
        * ``'float16'``: corresponding to 16-bit half precision consisting of
          10-bit for exponent and 5-bit for mantissa.
        * ``'bfloat16'``: corresponding to 16-bit half precision consisting
          of 7-bit for exponent and 8-bit for mantissa.
        * ``'float32'``: corresponding to 16-bit single precision consisting of
          23-bit for exponent and 8-bit for mantissa
        * ``'float64'``: corresponding to 64-bit double precision.
        * ``'float128'``: corresponding to 128-bit long double precision.

    See Also
    --------

    get_shape
    get_array_buffer

    Notes
    -----

    The code names ``'float8_e5m2'`` and ``'float8_e4m3'`` do not (yet) exist
    in data types. Including these names are experimental (more-or-less
    place-holders).
    """

    if 'float128' in array.dtype.__str__():
        data_type_name = b'float128'
    elif 'float64' in array.dtype.__str__():
        data_type_name = b'float64'
    elif 'float32' in array.dtype.__str__():
        data_type_name = b'float32'
    elif (('float16' in array.dtype.__str__()) and
          ('bfloat16' not in array.dtype.__str__())):
        data_type_name = b'float16'
    elif 'bfloat16' in array.dtype.__str__():
        data_type_name = b'bfloat16'
    elif 'float8_e5m2' in array.dtype.__str__():
        data_type_name = b'float8_e5m2'
    elif 'float8_e4m3' in array.dtype.__str__():
        data_type_name = b'float8_e4m3'
    else:
        raise ValueError('dtype is not recognized.')

    return data_type_name
