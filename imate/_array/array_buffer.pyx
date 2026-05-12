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
import warnings
from .array_properties import is_row_major
from cpython.buffer cimport PyBuffer_Release, PyObject_GetBuffer, \
    PyBUF_C_CONTIGUOUS, PyBUF_F_CONTIGUOUS

cdef extern from "Python.h":
    ctypedef long Py_intptr_t

__all__ = ['get_array_buffer', 'release_buffer']


# ======================
# get numpy array buffer
# ======================

cdef const void* _get_numpy_array_buffer(object array, Py_buffer* py_buffer):
    """
    Returns a C pointer to numpy's arrays buffer.

    Parameters
    ----------

    array : numpy.ndarray
        A numpy array

    Notes
    -----

    **Methods of getting buffer:**

    There are three other methods to get a pointer to the buffer of an array A:

    1. Using Cython's memoryview (dosn't work with float16, bflat16 types):

        cdef double[:, ::1] A_mv = A   # if A is a row-major array
        cdef double[::1, :] A_mv = A   # if A is a column-major array
        cdef double* A_ptr = &A[0, 0]  # from memoryview to C pointer

      This method does not work if the data type is __half, __nv_bfloat16, etc.
      This is because here we are reliant on Cython's memoryviews are strongly
      typed and only provide float, double and long double floatings.

    2. Using Numpy's API in Cython (works with any type):

        cimport numpy
        cdef numpy.ndarray A_npy = A  # Note: do not set data type
        cdef void* A_ptr = A_npy.data

      This method works with any type, such as float16 (which is __half in
      cuda), and bfloat16 (which is __nv_bfloat16 in cuda). However, it
      requires cimport-ing numpy, meaning that the Cyhton code should be
      compiled with numpy include dirs. That is, in setup.py:

        import numpy
        Extension(
            include_dirs=[..., numpy.get_include()],
        )

    3. Using CPython's API (this is what is implemented in this function).
       This method returns a void pointer, so it works for any data type.
       It also does not require numpy's headers at compile time. For more
       Cpython API usage, see: "cython/Cython/Include/cpython/buffer.pxd" in
       Cyhton's source code. Also see implementing buffer protocol in Cython:
       https://cython.readthedocs.io/en/latest/src/userguide/buffer.html

    See Also
    --------

    get_array_buffer
    """

    row_major = is_row_major(array)

    # Contiguity flag
    cdef int contiguity
    if row_major:
        contiguity = PyBUF_C_CONTIGUOUS
    else:
        contiguity = PyBUF_F_CONTIGUOUS

    # Request a buffer view of the array
    cdef int err = PyObject_GetBuffer(array, py_buffer, contiguity)

    if err != 0:
        raise RuntimeError('Cannot read buffer from Python object.')

    # Get a void pointer
    cdef const void* buffer = py_buffer.buf

    return buffer


# =======================
# get torch tensor buffer
# =======================

cdef const void* _get_torch_tensor_buffer(array):
    """
    Returns the C pointer of a torch.Tensor buffer.

    Parameters
    ----------

    array : numpy.ndarray or torch.Tensor or tensorflow.Tensor or JAX.Tensor
        An array

    Returns
    -------
    
    buffer : void*
        A void pointer of the buffer.

    See Also
    --------

    get_array_pointer
    """

    if not hasattr(array, 'data_ptr'):
        raise RuntimeError('Object is not a "torch.Tensor" type.')

    cdef Py_intptr_t int_ptr = array.data_ptr()
    cdef const void* buffer = <void*> int_ptr

    return buffer


# ================
# get array buffer
# ================

cdef const void* get_array_buffer(array, Py_buffer* py_buffer):
    """
    Returns the C pointer of an array.

    Parameters
    ----------

    array : numpy.ndarray or torch.Tensor or tensorflow.Tensor or JAX.Tensor
        An array

    py_buffer : Py_buffer*
        A pointer to a Py_buffer object (view).

    Returns
    -------

    Returns
    -------
    
    buffer : void*
        A void pointer of the buffer.

    See Also
    --------

    get_device
    get_shape
    is_row_major

    Notes
    -----

    Why py_buffer is needed:

    We only need the void* buffer pointer, which is the output of this
    function. However, this pointer is owned by py_buffer (since the buffer is
    the member py_buffer.buff). After working with the buffer pointer is done,
    the py_buffer object should be destroyed by calling PyBuffer_Release. This
    function should not be called while buffer is still in use. As such, we
    keep a reference to py_buffer (in py_c_linear_operator or
    py_cu_linear_object) and destroy py_buffer in the destructor of these
    object.

    Note that, calling PyBuffer_Release easly (while buffer is in use), in most
    cases does not harm the buffer, and the buffer can still be used. However,
    for the very specific case of using tensorflow tensors (when converted to
    numpy), by calling PyBuffer_Release, the buffer is destroyed. Hence, it is
    essential that for this specific case to keep py_buffer alive.
    """

    cdef const void* buffer = NULL

    if isinstance(array, numpy.ndarray):

        # Numpy array
        buffer = _get_numpy_array_buffer(array, py_buffer)

    elif ((type(array).__module__ == 'torch') and
          ('Tensor' in type(array).__name__ )):

        if array.device.type == 'gpu':
            warnings.warn('Tensor is on GPU; moving to CPU for buffer access.')
            array = array.cpu()

        buffer = _get_torch_tensor_buffer(array)

    elif (('tensorflow' in type(array).__module__) and
          ('Tensor' in type(array).__name__)):

        # Make sure array is on CPU
        import tensorflow as tf
        with tf.device('/CPU:0'):

            # Convert to numpy. If on GPU, this moves data from GPU to CPU. If
            # on CPU, the numpy is just a wrap without copying new data.
            array_tf = array.numpy()
            buffer = _get_numpy_array_buffer(array_tf, py_buffer)

    elif (('jaxlib' in type(array).__module__) and
          ('ArrayImpl' in type(array).__name__)):

        # Ensure any asynchronous operations are complete and on CPU
        array.block_until_ready()

        # From JAX tensor to numpy array
        array_jax = numpy.asarray(array) 
        buffer = _get_numpy_array_buffer(array_jax, py_buffer)

    else:
        raise RuntimeError('Matrix type is not supported.')

    return buffer


# ==============
# release buffer
# ==============

cdef void release_buffer(Py_buffer* py_buffer):
    """
    """

    # Destruct view object
    PyBuffer_Release(py_buffer)
