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

from ._device import get_processor_name, get_num_cpu_threads, get_gpu_name, \
        get_num_gpu_devices, get_nvidia_driver_version
from ._memory import Memory
from ._cuda import locate_cuda
from ..__version__ import __version__

__all__ = ['info']


# ====
# info
# ====

def info(print_only=True):
    """
    Provides general information about hardware device, package version, and
    memory usage.

    Parameters
    ----------

    print_only : bool, default=True
        It `True`, it prints the output. If `False`, it returns the output as
        a dictionary.

    Returns
    -------

    info_dict : dict
        (Only if ``print_only`` is `False`). A dictionary with the following
        keys:

        * ``imate_version``: `str`, the version of the imate package in the
          format ``"major_version.minor_version.patch_number"``.
        * ``processor``: `str`, the model name of the CPU processor.
        * ``num_threads``, `int`, number of CPU threads that are available and
          allocated to the user.
        * ``gpu_name``: `str`, model name of the GPU devices.
        * ``num_gpu_devices``: `int`, number of GPU devices in multi-GPU
          platforms.
        * ``cuda_version``: `str`, the version of CUDA Toolkit installed on the
          machine in the format ``"major_version.minor_version.patch_number"``.
        * ``nvidia_driver``: `str`, the version of NVIDIA graphic driver.
        * ``mem_used``: `int`, resident memory usage for the current Python
          process.
        * ``mem_unit``, `str`, the unit in which ``mem_used`` is reported. This
          can be ``"b"`` for Byte, ``"KB"`` for Kilo-Byte, ``"MB"`` for
          Mega-Byte, ``"GB"`` for Giga-Byte, and ``"TB"`` for Tera-Byte.

    See Also
    --------

    imate.device.get_processor_name
    imate.device.get_gpu_name
    imate.device.get_num_cpu_threads
    imate.device.get_num_gpu_devices
    imate.device.get_nvidia_driver_version
    imate.Memory
    imate.device.locate_cuda

    Notes
    -----

    **CUDA Version:**

    In order to find CUDA Toolkit information properly, either of the
    environment variables ``CUDA_HOME``, ``CUDA_ROOT``, or ``CUDA_PATH`` should
    be set to the directory where CUDA Toolkit is installed. Usually on UNIX
    operating systems, this path is ``/usr/local/cuda``. In this case, set
    ``CUDA_HOME`` (or any of the other variables mentioned in the above) as
    follows:

    ::

        export CUDA_HOME=/usr/local/cuda

    To permanently set this variable, place the above line in ``profile`` file,
    such as in ``~/.bashrc``, or ``~/.profile``, and source this file, for
    instance by

    ::

        source ~/.bashrc

    If no CUDA Toolkit is installed, then the key ``cuda_version`` shows
    `not found`.

    .. note::

        It is possible that the CUDA Toolkit is installed on the machine, but
        ``cuda_version`` key shows `not found`. This is because the user did
        not set the environment variables mentioned in the above.

    **GPU Devices:**

    If the key ``gpu_name`` shows `not found`, this is because either

    * No GPU device is detected on the machine.
    * GPU device exists, but NVIDIA graphic driver is not installed. See
      :ref:`Install NVIDIA Graphic Driver <install-graphic-driver>` for further
      details.
    * NVIDIA graphic driver is installed, but the executable ``nvidia-smi`` is
      not available on the `PATH``. To fix this, set the location of the
      ``nvidia-smi`` executable on the ``PATH`` variable.

    **Memory:**

    The key ``mem_used`` shows the resident set size memory (RSS) on RAM
    hardware. The unit of the reported memory size can be found in
    ``mem_unit``, which can be ``b`` for Bytes, ``KB`` for Kilo-Bytes, ``MB``
    for Mega-Bytes, and so on.

    Examples
    --------

    Print information:

    .. code-block:: python

        >>> from imate import info
        >>> info()
        imate version   : 0.13.0
        processor       : Intel(R) Xeon(R) CPU E5-2623 v3 @ 3.00GHz
        num threads     : 8
        gpu device      : 'GeForce GTX 1080 Ti'
        num gpu devices : 4
        cuda version    : 11.2.0
        nvidia driver   : 460.84
        process memory  : 1.7 (Gb)

    Return information as a dictionary:

    .. code-block:: python

        >>> from imate import info
        >>> info_dict = info(print_only=False)

        >>> # Neatly print dictionary using pprint
        >>> from pprint import pprint
        >>> pprint(info_dict)
        {
            'imate version': 0.13.0,
            'processor': Intel(R) Xeon(R) CPU E5-2623 v3 @ 3.00GHz,
            'num threads': 8,
            'gpu device': 'GeForce GTX 1080 Ti',
            'num gpu devices': 4,
            'cuda version': 11.2.0,
            'nvidia driver': 460.84,
            'process memory': 1.7 (Gb)
        }
    """

    mem_used, mem_unit = Memory.get_resident_memory(human_readable=True)

    # Get cuda version
    cuda = locate_cuda()
    if cuda != {}:
        cuda_version = cuda['version']
        cuda_version_ = '%d.%d.%d' \
            % (cuda_version['major'], cuda_version['minor'],
               cuda_version['patch'])
    else:
        cuda_version_ = 'not found'

    # NVIDIA driver version
    nvidia_driver = get_nvidia_driver_version()

    info_ = {
        'imate_version': __version__,
        'processor_name': get_processor_name(),
        'num_cpu_threads': get_num_cpu_threads(),
        'gpu_name': get_gpu_name(),
        'num_gpu_devices': get_num_gpu_devices(),
        'mem_used': mem_used,
        'mem_unit': mem_unit,
        'cuda_version': cuda_version_,
        'nvidia_driver': nvidia_driver,
    }

    # Print
    if print_only:
        print('')
        print('imate version   : %s' % info_['imate_version'])
        print('processor       : %s' % info_['processor_name'])
        print('num threads     : %d' % info_['num_cpu_threads'])
        print('gpu device      : %s' % info_['gpu_name'])
        print('num gpu devices : %d' % info_['num_gpu_devices'])
        print('cuda version    : %s' % info_['cuda_version'])
        print('nvidia driver   : %s' % info_['nvidia_driver'])
        print('process memory  : %0.1f (%s)'
              % (info_['mem_used'], info_['mem_unit']))
        print('')
    else:
        return info_
