.. _inquiry-gpu:

Inquiry GPU and CUDA with |project|
===================================

First, make sure |project| recognizes the CUDA libraries and GPU device. There are a number of functions available in :ref:`imate.device <Device Inquiry>` module to inquiry GPU device.

Locate CUDA Toolkit
-------------------

Use :func:`imate.device.locate_cuda` function to find the location of CUDA home directory.

.. code-block:: python

    >>> import imate

    >>> # Get the location and version of CUDA Toolkit
    >>> imate.device.locate_cuda()
    {
        'home': '/global/software/sl-7.x86_64/modules/langs/cuda/11.2',
        'include': '/global/software/sl-7.x86_64/modules/langs/cuda/11.2/include',
        'lib': '/global/software/sl-7.x86_64/modules/langs/cuda/11.2/lib64',
        'nvcc': '/global/software/sl-7.x86_64/modules/langs/cuda/11.2/bin/nvcc',
        'version':
        {
            'major': 11,
            'minor': 2,
            'patch': 0
        }
    }

If the above function does not return an output such as in the above, it is because either `CUDA Toolkit` is not installed, or the directory of the CUDA Toolkit is not set. To do so, set the directory of CUDA Toolkit to either of the variables ``CUDA_HOME``, ``CUDA_ROOT``, or ``CUDA_PATH``, such as by

.. prompt:: bash

    echo 'export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}' >> ~/.bashrc
    echo 'export CUDA_HOME=/usr/local/cuda${CUDA_HOME:+:${CUDA_HOME}}' >> ~/.bashrc
    source ~/.bashrc

Detect NVIDIA Graphic Driver
----------------------------

Use :func:`imate.device.get_nvidia_driver_version` function to make sure |project| can detect the NVIDIA driver.

.. code-block:: python

    >>> # Get the version of NVIDIA graphic driver
    >>> imate.device.get_nvidia_driver_version()
    460.84

Detect GPU Devices
------------------

Use :func:`imate.device.get_processor_name()` and :func:`imate.device.get_gpu_name()` to find the name of CPU and GPU devices, respectively.

.. code-block:: python

    >>> # Get the name of CPU processor
    >>> imate.device.get_processor_name()
    'Intel(R) Xeon(R) CPU E5-2623 v3 @ 3.00GHz'

    >>> # Get the name of GPU devices
    >>> imate.device.get_gpu_name()
    'GeForce GTX 1080 Ti'

.. note::

    If the name of the GPU device is empty, this is because either there is no GPU device detected, or *NVIDIA graphic driver* is not installed, or its location is not on the PATH. To do so, set the location of ``nvidia-smi`` executable to the ``PATH`` environment variable. On UNIX, this executable should be on ``/usr/bin`` directory and by default it should be already on the `PATH`.

The number of CPU threads and GPU devices can be obtained respectively by :func:`imate.device.get_num_cpu_threads` and :func:`imate.device.get_num_gpu_devices()` functions.

.. code-block:: python

    >>> # Get number of processor threads
    >>> imate.device.get_num_cpu_threads()
    8

    >>> # Get number of GPU devices
    >>> imate.device.get_num_gpu_devices()
    4

The :func:`imate.info` function also obtains general information about |project| configuration and devices.

.. code-block:: python
   :emphasize-lines: 5, 6, 7, 8

    >>> imate.info()
    imate version   : 0.13.0
    processor       : Intel(R) Xeon(R) CPU E5-2623 v3 @ 3.00GHz
    num threads     : 8
    gpu device      : 'GeForce GTX 1080 Ti'
    num gpu devices : 4
    cuda version    : 11.2.0
    nvidia driver   : 460.84
    process memory  : 1.7 (Gb)

Alternatively, one may directly use ``nvidia-smi`` command to inquiry the GPU devices.

.. prompt:: bash

    nvidia-smi

Output:

.. code-block:: Text

    +-----------------------------------------------------------------------------+
    | NVIDIA-SMI 460.84       Driver Version: 460.84       CUDA Version: 11.2     |
    |-------------------------------+----------------------+----------------------+
    | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
    | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
    |                               |                      |               MIG M. |
    |===============================+======================+======================|
    |   0  GeForce GTX 108...  Off  | 00000000:02:00.0 Off |                  N/A |
    | 33%   57C    P2    62W / 250W |    147MiB / 11178MiB |     25%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+
    |   1  GeForce GTX 108...  Off  | 00000000:03:00.0 Off |                  N/A |
    | 27%   48C    P2    61W / 250W |    147MiB / 11178MiB |     23%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+
    |   2  GeForce GTX 108...  Off  | 00000000:81:00.0 Off |                  N/A |
    | 18%   32C    P0    59W / 250W |      0MiB / 11178MiB |      0%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+
    |   3  GeForce GTX 108...  Off  | 00000000:82:00.0 Off |                  N/A |
    | 18%   32C    P0    59W / 250W |      0MiB / 11178MiB |      0%      Default |
    |                               |                      |                  N/A |
    +-------------------------------+----------------------+----------------------+
    
    +-----------------------------------------------------------------------------+
    | Processes:                                                                  |
    |  GPU   GI   CI        PID   Type   Process name                  GPU Memory |
    |        ID   ID                                                   Usage      |
    |=============================================================================|
    |    0   N/A  N/A       654      C   python                            145MiB |
    |    1   N/A  N/A       839      C   python                            145MiB |
    +-----------------------------------------------------------------------------+

The output of ``nvidia-smi`` in the above shows there are four GPU devices available on the machine. For more complete information on the GPU devices, use

.. prompt:: bash

    nvidia-smi -q
