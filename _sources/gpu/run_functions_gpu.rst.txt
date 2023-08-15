.. _run-functions-gpu:

Run |project| Functions on GPU
==============================

All functions in |project| that accept the `SLQ` method (using ``method=slq`` argument) can perform computations on GPU devices. To do so, include ``gpu=True`` argument to the function syntax. The following examples show using multi-GPU devices to compute the log-determinant of a large matrix.

A Simple Example
----------------

First, create a sample Toeplitz matrix with ten million in size using :func:`imate.toeplitz` function.

.. code-block:: python

    >>> # Import toeplitz matrix
    >>> from imate import toeplitz

    >>> # Generate a sample matrix (a toeplitz matrix)
    >>> n = 10000000
    >>> A = toeplitz(2, 1, size=n, gram=True)

Next, create an :class:`imate.Matrix` object from matrix `A`:

.. code-block:: python

    >>> # Import Matrix class
    >>> from imate import Matrix

    >>> # Create a matrix operator object from matrix A
    >>> Aop = Matrix(A)

Compute the log-determinant of the above matrix on GPU by passing ``gpu=True`` to :func:`imate.logdet` function. Recall GPU can only be employed using `SLQ` method by passing ``method=slq`` argument.

.. code-block:: python
    :emphasize-lines: 5

    >>> # Import logdet function
    >>> from imate import logdet

    >>> # Compute log-determinant of Aop
    >>> logdet(Aop, method='slq', gpu=True)
    13862193.020813728

Get Process Information
-----------------------

It is useful pass the argument ``return_info=True`` to get information about the computation process.

.. code-block:: python

    >>> # Compute log-determinant of Aop
    >>> ld, info = logdet(Aop, method='slq', gpu=True, return_info=True)

The information about GPU devices used during the computation can be found in ``info['device']`` key:

.. code-block:: python
    :emphasize-lines: 5, 6, 7

    >>> from pprint import pprint
    >>> pprint(info['device'])
    {
        'num_cpu_threads': 8,
        'num_gpu_devices': 4,
        'num_gpu_multiprocessors': 28,
        'num_gpu_threads_per_multiprocessor': 2048
    }

The processing time can be obtained by ``info['time']`` key:

.. code-block:: python

    >>> pprint(info['time'])
    {
        'alg_wall_time': 1.7192635536193848,
        'cpu_proc_time': 3.275628339,
        'tot_wall_time': 3.5191736351698637
    }

Verbose Output
--------------

Alternatively, to print verbose information, including the information about GPU devices, pass ``verbose=True`` to the function argument:

.. code-block:: python

    >>> # Compute log-determinant of Aop
    >>> logdet(Aop, method='slq', gpu=True, verbose=True)

The above script prints the following table. The last section of the table shows device information.

.. literalinclude:: ../_static/data/imate.logdet.slq-verbose-gpu.txt
    :language: python
    :emphasize-lines: 28, 29, 30

Set Number of GPU Devices
-------------------------

By default, |project| employs the maximum number of available GPU devices. To employ a specific number of GPU devices, set ``num_gpu-devices`` in the function arguments. For instance

.. code-block:: python
    :emphasize-lines: 6, 12

    >>> # Import logdet function
    >>> from imate import logdet

    >>> # Compute log-determinant of Aop
    >>> ld, info = logdet(Aop, method='slq', gpu=True, return_info=True,
    ...                   num_gpu_devices=2)

    >>> # Check how many GPU devices used
    >>> pprint(info['device'])
    {
        'num_cpu_threads': 8,
        'num_gpu_devices': 2,
        'num_gpu_multiprocessors': 28,
        'num_gpu_threads_per_multiprocessor': 2048
    }
