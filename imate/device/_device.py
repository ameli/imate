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

import os
import re
import platform
import subprocess

__all__ = ['get_processor_name', 'get_num_cpu_threads', 'get_gpu_name',
           'get_num_gpu_devices', 'get_nvidia_driver_version',
           'restrict_to_single_processor']


# ==================
# get processor name
# ==================

def get_processor_name():
    """
    Gets the model name of CPU processor.

    Returns
    -------

    gpu_name : str
        Processor name

    See Also
    --------

    imate.device.get_num_cpu_threads
    imate.device.get_gpu_name
    imate.info

    Notes
    -----

    For `Linux`, this function parses the output of

    ::

        cat /proc/cpuino

    For `macOS`, this function parses the output of

    ::

        sysctl -n machdep.cpu.brand_string

    .. warning::

        For Windows operating system, this function does not get the full brand
        name of the cpu processor

    Examples
    --------

    .. code-block:: python

        >>> from imate.device import get_processor_name
        >>> get_processor_name()
        'Intel(R) Xeon(R) CPU E5-2623 v3 @ 3.00GHz'
    """

    if platform.system() == "Windows":
        return platform.processor()

    elif platform.system() == "Darwin":
        os.environ['PATH'] = os.environ['PATH'] + os.pathsep + '/usr/sbin'
        command = "sysctl -n machdep.cpu.brand_string"
        return subprocess.getoutput(command).strip()

    elif platform.system() == "Linux":
        command = "cat /proc/cpuinfo"
        all_info = subprocess.getoutput(command).strip()
        for line in all_info.split("\n"):
            if "model name" in line:
                return re.sub(".*model name.*:", "", line, 1)[1:]

    return ""


# ===================
# get num cpu threads
# ===================

def get_num_cpu_threads():
    """
    Returns the number of available CPU processor threads.

    Returns
    -------

    num_threads : int
        Number of processor threads.

    See Also
    --------

    imate.device.get_processor_name
    imate.device.get_num_gpu_devices
    imate.info

    Notes
    -----

    The returned value is not the total number of CPU threads. Rather, the
    number of available CPU threads that is allocated to the user is returned.
    For instance, if on a device with `8` threads, only `2` threads are
    allocated to a user, the return value of this function is `2`.

    Examples
    --------

    Find the number of *available* CPU threads:

    .. code-block:: python

        >>> from imate.device import get_num_cpu_threads
        >>> get_num_cpu_threads()
        2

    Find the total number of CPU threads on the device (all may not be
    available/allocated to the user):

    .. code-block:: python

        >>> # Method 1
        >>> import os
        >>> os.cpu_count()
        8

        >>> # Method 2
        >>> import multiprocessing
        >>> multiprocessing.cpu_count()
        8
    """

    if hasattr(os, 'sched_getaffinity'):
        num_cpu_threads = len(os.sched_getaffinity(0))
    else:
        num_cpu_threads = os.cpu_count()

    return num_cpu_threads


# ============
# get gpu name
# ============

def get_gpu_name():
    """
    Gets the model name of GPU device.

    Returns
    -------

    cpu_name : str
        GPU model name. If no GPU device is found, it returns ``"not found"``.

    See Also
    --------

    imate.device.get_num_gpu_devices
    imate.device.get_processor_name
    imate.info

    Notes
    -----

    This function parses the output of ``nvidia-smi`` command as

    ::

        nvidia-smi -a | grep -i "Product Name" -m 1 | grep -o ":.*" |cut -c 3-

    The ``nvidia-smi`` command is part of `NVIDIA graphic driver`. See
    :ref:`Install NVIDIA Graphic Driver <install-graphic-driver>` for further
    details. If a graphic driver is not installed, this function returns
    ``"not found"``.

    Examples
    --------

    .. code-block:: python

        >>> from imate.device import get_gpu_name
        >>> get_gpu_name()
        'GeForce GTX 1080 Ti'
    """

    # Pre-check if the nvidia-smi command works
    status = _check_nvidia_smi()
    if status is False:
        gpu_name = 'not found'
        return gpu_name

    command = 'nvidia-smi -a | grep -i "Product Name" -m 1 | grep -o ":.*"' + \
        ' | cut -c 3-'

    try:
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        stdout, _ = process.communicate()
        error_code = process.poll()

        # Error code 127 means nvidia-smi is not a recognized command. Error
        # code 9 means nvidia-smi could not find any device.
        if error_code != 0:
            gpu_name = 'not found'
        else:
            gpu_name = stdout.strip().decode('utf-8')

        if gpu_name == '':
            gpu_name = 'not found'

    except FileNotFoundError:
        gpu_name = 'not found'

    return gpu_name


# ===================
# get num gpu devices
# ===================

def get_num_gpu_devices():
    """
    Returns the number of available GPU devices in multi-GPU platforms.

    Returns
    -------

    num_gpu : int
        Number of GPU devices.

    See Also
    --------

    imate.device.get_gpu_name
    imate.device.get_num_cpu_threads
    imate.info

    Notes
    -----

    The returned value is not the total number of GPU devices on the machine.
    Rather, the number of available GPU devices that is allocated to the user
    is returned.

    This function parses the output of ``nvidia-smi`` command by

    ::

        nvidia-smi --list-gpus | wc -l

    The ``nvidia-smi`` command is part of `NVIDIA graphic driver`. See
    :ref:`Install NVIDIA Graphic Driver <install-graphic-driver>` for further
    details. If a graphic driver is not installed, this function returns `0`.

    Examples
    --------

    .. code-block:: python

        >>> from imate.device import get_num_gpu_devices
        >>> get_num_gpu_devices()
        4
    """

    # Pre-check if the nvidia-smi command works
    status = _check_nvidia_smi()
    if status is False:
        num_gpu_devices = 0
        return num_gpu_devices

    command = 'nvidia-smi --list-gpus | wc -l'

    try:
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        stdout, _ = process.communicate()
        error_code = process.poll()

        # Error code 127 means nvidia-smi is not a recognized command. Error
        # code 9 means nvidia-smi could not find any device.
        if error_code != 0:
            num_gpu_devices = 0
        else:
            num_gpu_devices = int(stdout)

    except FileNotFoundError:
        num_gpu_devices = 0

    return num_gpu_devices


# =========================
# get nvidia driver version
# =========================

def get_nvidia_driver_version():
    """
    Gets the NVIDIA graphic driver version.

    Returns
    -------

    version : str
        The version number in the format "DriverVersion.RuntimeVersion".

    See Also
    --------

    imate.device.locate_cuda
    imate.info

    Notes
    -----

    This function parses the output of ``nvidia-smi`` command as

    ::

        nvidia-smi -q | grep -i "Driver Version" | grep -o ":.*" | cut -c 3-

    The ``nvidia-smi`` command is part of `NVIDIA graphic driver`. See
    :ref:`Install NVIDIA Graphic Driver <install-graphic-driver>` for further
    details. If a graphic driver is not installed, this function returns
    ``"not found"``.

    Examples
    --------

    .. code-block:: python

        >>> from imate.device import get_nvidia_driver_version
        >>> get_nvidia_driver_version()
        '460.84'
    """

    # Pre-check if the nvidia-smi command works
    status = _check_nvidia_smi()
    if status is False:
        gpu_name = 'not found'
        return gpu_name

    command = 'nvidia-smi -q | grep -i "Driver Version" | grep -o ":.*"' + \
        '| cut -c 3-'

    try:
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        stdout, _ = process.communicate()
        error_code = process.poll()

        # Error code 127 means nvidia-smi is not a recognized command. Error
        # code 9 means nvidia-smi could not find any device.
        if error_code != 0:
            version = 'not found'
        else:
            version = stdout.strip().decode('utf-8')

        if version == '':
            version = 'not found'

    except FileNotFoundError:
        version = 'not found'

    return version


# ================
# check nvidia smi
# ================

def _check_nvidia_smi():
    """
    Checks if the ``nvidia-smi`` command can connect to the NVIDIA device.

    Returns
    -------

    status : bool
        If `True`, the output of ``nvidia-smi`` command is successful.
    """

    command = 'nvidia-smi'
    status = True

    try:
        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE)
        stdout, _ = process.communicate()
        error_code = process.poll()

        # Error code 127 means nvidia-smi is not a recognized command. Error
        # code 9 means nvidia-smi could not find any device.
        if error_code != 0:
            status = False

    except FileNotFoundError:
        status = False

    return status


# ============================
# restrict to single processor
# ============================

def restrict_to_single_processor():
    """
    Restricts the computations to only one CPU thread.

    This function is primarily used to properly measure the process time of a
    computational task using ``time.process_time`` from ``time`` module.

    .. note::

        This function should be called *at the very first line of the main
        script of your Python code* before importing any other Python package.

    See Also
    --------

    imate.device.get_num_cpu_threads
    imate.info

    Notes
    -----

    **Why using this function:**

    In Python, to measure the CPU `processing time` (and not the `wall time`)
    of a computational task, the function ``time.process_time`` from ``time``
    module can be used. For instance

    .. code-block:: python

        >>> import time

        >>> t_init = time.process_time()

        >>> # Perform some time-consuming task

        >>> t_final = time.process_time()
        >>> t_process = t_final - t_init

    The process time differs from wall time in which it measures the total
    process time of all CPU threads and excludes the idle times when the
    process was not working.

    Often, measuring the process time is affected by other factors and the
    result of the above approach is not reliable. One example of such is
    measuring the process time of the global optimization problem using
    ``scipy.optimize.differential_evolution`` function:

    .. code-block:: python

        >>> from scipy.optimize import differential_evolution
        >>> import time

        >>> t_init = time.process_time()
        >>> result = differential_evolution(worker=num_workers, ...)
        >>> t_final = time.process_time()
        >>> t_process = t_final - t_init

    However, regardless of setting ``worker=1``, or ``worker=-1``, the measured
    process time is identical, hence cannot be trusted.

    A solution to this problem is to restrict the computational task to use
    only one CPU thread.

    **Alternative Solution:**

    Instead of calling this function, export the following environment
    variables *before* executing your Python script:

    .. code-block:: bash

        export OMP_NUM_THREADS=1
        export OPENBLAS_NUM_THREADS=1
        export MKL_NUM_THREADS=1
        export VECLIB_MAXIMUM_THREADS=1
        export NUMEXPR_NUM_THREADS=1

    Examples
    --------

    .. code-block:: python

        >>> # Call this function before importing any other module
        >>> from imate.device import restrict_to_single_processor
        >>> restrict_to_single_processor()

        >>> # Import packages
        >>> import scipy, numpy, imate, time
    """

    # Uncomment lines below if measuring elapsed time. These will restrict
    # python to only use one processing thread.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"
