.. _imate-gpu:

Using GPU Devices
*****************

.. contents::

|project| can run on `CUDA-capable` GPU devices with the following installed:

1. NVIDIA graphic driver,
2. CUDA libraries.

.. rubric:: CUDA Version

The version of CUDA libraries installed on the user's machine should match the version of the CUDA libraries that |project| package was compiled with. This includes matching both *major* and *minor* parts of the version numbers. However, the version's *patch* numbers do not need to be matched.

.. note::

    The |project| package that is installed with either ``pip`` or ``conda`` already has built-in support for CUDA Toolkit. The latest version of |project| is compatible with **CUDA 11.7.x**, which should match the CUDA version installed on the user's machine.

.. topic:: Methods of Setting up CUDA and |project|

    There are three ways to use |project| with a compatible version of CUDA Toolkit:

    1. :ref:`Install NVIDIA CUDA Toolkit <gpu-install-cuda>` with a CUDA version compatible with an existing |project| installation. In this way, you can keep the |project| package that is already installed with ``pip`` or ``conda``.
    2. :ref:`Compile imate from the source <gpu-compile-imate>` for a specific version of CUDA to use an existing CUDA library. In this way, you can keep the current CUDA installation.
    3. :ref:`Use docker image <gpu-docker>` with pre-installed |project|, CUDA libraries, and NVIDIA graphic driver. This is the most convenient way as no compilation or installation of |project| and CUDA Toolkit is required.

The above methods are described in order below.

.. _gpu-install-cuda:

Install NVIDIA CUDA Toolkit
===========================

The following instruction describes installing `CUDA 11.7` for `Ubuntu 22.04`, `CentOS 7`, and `Red Hat 9 (RHEL 9)` on the `X86_64` platform. You may refer to `CUDA installation guide <https://developer.nvidia.com/cuda-downloads>`_ from NVIDIA developer documentation for other operating systems and platforms.

.. attention::

    NVIDIA does not support macOS. You can install the NVIDIA CUDA Toolkit on Linux and Windows only.

.. _install-graphic-driver:

Install NVIDIA Graphic Driver
-----------------------------

Register NVIDIA CUDA repository by

.. tab-set::

    .. tab-item:: Ubuntu/Debian
        :sync: ubuntu

        .. prompt:: bash

            wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
            sudo dpkg -i cuda-keyring_1.0-1_all.deb
            sudo apt update

    .. tab-item:: CentOS 7
        :sync: centos

        .. prompt:: bash

            sudo yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo
            sudo yum clean all

    .. tab-item:: RHEL 9
        :sync: rhel

        .. prompt:: bash

            sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel9/x86_64/cuda-rhel9.repo
            sudo dnf clean all

Install *NVIDIA graphic driver* with

.. tab-set::

    .. tab-item:: Ubuntu/Debian
        :sync: ubuntu

        .. prompt:: bash

            export DEBIAN_FRONTEND=noninteractive
            sudo -E apt install cuda-drivers -y

    .. tab-item:: CentOS 7
        :sync: centos

        .. prompt:: bash

            sudo yum -y install nvidia-driver-latest-dkms

    .. tab-item:: RHEL 9
        :sync: rhel

        .. prompt:: bash

            sudo dnf -y module install nvidia-driver:latest-dkms

The above step might need a *reboot* afterwards to properly load NVIDIA graphic driver. Confirm the driver installation by

.. prompt:: bash

   nvidia-smi

.. _install-cuda-toolkit:

Install CUDA Toolkit
--------------------

It is not required to install the entire CUDA Toolkit (2.6GB). Rather, only the CUDA runtime library, cuBLAS, and cuSparse libraries are sufficient (700MB in total). These can be installed by

.. tab-set::

    .. tab-item:: Ubuntu/Debian
        :sync: ubuntu

        .. prompt:: bash
           
           sudo apt install cuda-cudart-11-7 libcublas-11-7 libcusparse-11-7 -y

    .. tab-item:: CentOS 7
        :sync: centos

        .. prompt:: bash

           sudo yum install --setopt=obsoletes=0 -y \
                cuda-nvcc-11-7.x86_64 \
                libcublas-11-7.x86_64 \
                libcusparse-11-7.x86_64

    .. tab-item:: RHEL 9
        :sync: rhel

        .. prompt:: bash

           sudo dnf install --setopt=obsoletes=0 -y \
                cuda-nvcc-11-7.x86_64 \
                libcublas-11-7.x86_64 \
                libcusparse-11-7.x86_64

Update ``PATH`` with the CUDA installation location by

.. prompt:: bash

    echo 'export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}' >> ~/.bashrc
    echo 'export CUDA_HOME=/usr/local/cuda${CUDA_HOME:+:${CUDA_HOME}}' >> ~/.bashrc
    source ~/.bashrc

Install OpenMP
--------------

In addition to CUDA Toolkit, make sure the `OpenMP` library is also installed using

.. tab-set::

    .. tab-item:: Ubuntu/Debian
        :sync: ubuntu

        .. prompt:: bash

            sudo apt install libgomp1 -y

    .. tab-item:: CentOS 7
        :sync: centos

        .. prompt:: bash

            sudo yum install libgomp -y

    .. tab-item:: RHEL 9
        :sync: rhel

        .. prompt:: bash

            sudo dnf install libgomp -y

.. _gpu-compile-imate:

Compile |project| from Source with CUDA
=======================================

Install C++ Compiler and OpenMP
-------------------------------

Compile |project| with either of GCC, Clang/LLVM, or Intel C++ compiler.

.. rubric:: Install GNU GCC Compiler

.. tab-set::

    .. tab-item:: Ubuntu/Debian
        :sync: ubuntu

        .. prompt:: bash

            sudo apt install build-essential libomp-dev

    .. tab-item:: CentOS 7
        :sync: centos

        .. prompt:: bash

            sudo yum group install "Development Tools"

    .. tab-item:: RHEL 9
        :sync: rhel

        .. prompt:: bash

            sudo dnf group install "Development Tools"

Then, export ``C`` and ``CXX`` variables by

.. prompt:: bash

  export CC=/usr/local/bin/gcc
  export CXX=/usr/local/bin/g++

.. rubric:: Install Clang/LLVN Compiler
  
.. tab-set::

    .. tab-item:: Ubuntu/Debian
        :sync: ubuntu

        .. prompt:: bash

            sudo apt install clang libomp-dev

    .. tab-item:: CentOS 7
        :sync: centos

        .. prompt:: bash

            sudo yum install yum-utils
            sudo yum-config-manager --enable extras
            sudo yum makecache
            sudo yum install clang

    .. tab-item:: RHEL 9
        :sync: rhel

        .. prompt:: bash

            sudo dnf install yum-utils
            sudo dnf config-manager --enable extras
            sudo dnf makecache
            sudo dnf install clang

Then, export ``C`` and ``CXX`` variables by

.. prompt:: bash

  export CC=/usr/local/bin/clang
  export CXX=/usr/local/bin/clang++

.. rubric:: Install Intel oneAPI Compiler

To install `Intel Compiler` see `Intel oneAPI Base Toolkit <https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html?operatingsystem=linux&distributions=aptpackagemanager>`_.

Install CUDA Compiler and Development Libraries
-----------------------------------------------

.. attention::

    The minimum version of CUDA to compile |project| is `CUDA 10.0`.

If CUDA Toolkit is installed, skip this part. Otherwise, Make sure the CUDA compiler and the development libraries of cuBLAS and cuSparse are installed by

.. tab-set::

    .. tab-item:: Ubuntu/Debian
        :sync: ubuntu

        .. prompt:: bash

            sudo apt install -y \
                cuda-nvcc-11-7 \
                libcublas-11-7 \
                libcublas-dev-11-7 \
                libcusparse-11-7 -y \
                libcusparse-dev-11-7

    .. tab-item:: CentOS 7
        :sync: centos

        .. prompt:: bash

            sudo yum install --setopt=obsoletes=0 -y \
                cuda-nvcc-11-7.x86_64 \
                cuda-cudart-devel-11-7.x86_64 \
                libcublas-11-7.x86_64 \
                libcublas-devel-11-7.x86_64 \
                libcusparse-11-7.x86_64 \
                libcusparse-devel-11-7.x86_64

    .. tab-item:: RHEL 9
        :sync: rhel

        .. prompt:: bash

            sudo dnf install --setopt=obsoletes=0 -y \
                cuda-nvcc-11-7.x86_64 \
                cuda-cudart-devel-11-7.x86_64 \
                libcublas-11-7.x86_64 \
                libcublas-devel-11-7.x86_64 \
                libcusparse-11-7.x86_64 \
                libcusparse-devel-11-7.x86_64

Update ``PATH`` with the CUDA installation location by

.. prompt:: bash

    echo 'export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}' >> ~/.bashrc
    source ~/.bashrc

Check if the CUDA compiler is available with ``which nvcc``.

Load CUDA Compiler on GPU Cluster
---------------------------------

If you are compiling |project| on a GPU cluster, chances are the CUDA Toolkit is already installed. If the cluster uses ``module`` interface, load CUDA as follows.

First, check if a CUDA module is available by

.. prompt:: bash

    module avail

Load both CUDA and GCC by

.. prompt:: bash

    module load cuda gcc

You may specify CUDA version if multiple CUDA versions are available, such as by

.. prompt:: bash

    module load cuda/11.7 gcc/6.3

You may check if CUDA Compiler is available with ``which nvcc``.

Configure Compile-Time Environment Variables
--------------------------------------------

Specify the home directory of CUDA Toolkit by setting either of the variables ``CUDA_HOME``, ``CUDA_ROOT``, or ``CUDA_PATH``. The home directory should be a path containing the executable ``/bin/nvcc``. For instance, if ``/usr/local/cuda/bin/nvcc`` exists, export the following:

.. prompt:: bash

    export CUDA_HOME=/usr/local/cuda

To permanently set this variable, place the above line in a profile file, such as in ``~/.bashrc``, or ``~/.profile``, and source this file, for instance by

.. prompt:: bash

    echo 'export CUDA_HOME=/usr/local/cuda${CUDA_HOME:+:${CUDA_HOME}}' >> ~/.bashrc
    source ~/.bashrc

To compile |project| with CUDA, export the following flag variable

.. prompt:: bash

    export USE_CUDA=1

Enable Dynamic Loading (*optional*)
-----------------------------------

When |project| is complied, the CUDA libraries bundle with the final installation of |project| package, making it over 700MB. While this is generally not an issue for most users, often a small package is preferable if the installed package has to be distributed to other machines. To this end, enable the `dynamic loading` feature of |project|. In this case, the CUDA libraries do not bundle with the |project| installation, rather, |project| loads the existing CUDA libraries of the host machine at runtime. To enable dynamic loading, simply set:

.. prompt:: bash
    
    export CUDA_DYNAMIC_LOADING=1

Compile and Install
-------------------

|repo-size|

Get the source code of |project| with

.. prompt:: bash

    git clone https://github.com/ameli/imate.git

Compile and install by

.. prompt:: bash

    cd imate
    python setup.py install

.. _gpu-docker:

Use |project| Docker Container on GPU
=====================================
   
This method neither requires installing CUDA nor |project| as all are pre-installed in a docker image.

Install Docker
--------------

First, `install docker <https://docs.docker.com/engine/install/ubuntu/>`_. Briefly:

.. tab-set::

    .. tab-item:: Ubuntu/Debian
        :sync: ubuntu

        .. prompt:: bash

            sudo apt update
            sudo apt install ca-certificates curl gnupg lsb-release
            sudo mkdir -p /etc/apt/keyrings
            curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
                sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
            echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
                https://download.docker.com/linux/ubuntu \
                $(lsb_release -cs) stable" | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
            sudo apt update
            sudo apt install docker-ce docker-ce-cli containerd.io docker-compose-plugin

    .. tab-item:: CentOS 7
        :sync: centos

        .. prompt:: bash

            sudo yum install -y yum-utils
            sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
            sudo yum install docker-ce docker-ce-cli containerd.io docker-compose-plugin
            sudo systemctl enable docker.service
            sudo systemctl enable containerd.service
            sudo systemctl start docker

    .. tab-item:: RHEL 9
        :sync: rhel

        .. prompt:: bash

            sudo yum install -y yum-utils
            sudo yum-config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
            sudo yum install docker-ce docker-ce-cli containerd.io docker-compose-plugin
            sudo systemctl enable docker.service
            sudo systemctl enable containerd.service
            sudo systemctl start docker

Configure docker to run docker `without sudo password <https://docs.docker.com/engine/install/linux-postinstall/>`_ by

.. prompt:: bash

    sudo groupadd docker
    sudo usermod -aG docker $USER

Then, log out and log back. If docker is installed on a *virtual machine*, restart the virtual machine for changes to take effect.

Install NVIDIA Container Toolkit
--------------------------------

To access host's GPU device from a docker container, `install NVIDIA Container Toolkit <https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html>`_ as follows.

Add the package to the repository:

.. tab-set::

    .. tab-item:: Ubuntu/Debian
        :sync: ubuntu

        .. prompt:: bash

            distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
            curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
            curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

    .. tab-item:: CentOS 7
        :sync: centos

        .. prompt:: bash

            sudo yum-config-manager --add-repo=https://download.docker.com/linux/centos/docker-ce.repo

    .. tab-item:: RHEL 9
        :sync: rhel

        .. prompt:: bash

            sudo dnf config-manager --add-repo=https://download.docker.com/linux/centos/docker-ce.repo

Install `nvidia-contaner-toolkit` by:

.. tab-set::

    .. tab-item:: Ubuntu/Debian
        :sync: ubuntu

        .. prompt:: bash

            sudo apt update
            sudo apt install -y nvidia-container-toolkit

    .. tab-item:: CentOS 7
        :sync: centos

        .. prompt:: bash

            sudo yum install -y https://download.docker.com/linux/centos/7/x86_64/stable/Packages/containerd.io-1.4.3-3.1.el7.x86_64.rpm

    .. tab-item:: RHEL 9
        :sync: rhel

        .. prompt:: bash

            sudo dnf install -y https://download.docker.com/linux/centos/7/x86_64/stable/Packages/containerd.io-1.4.3-3.1.el7.x86_64.rpm

Restart docker:

.. prompt:: bash

    sudo systemctl restart docker

Get |project| Docker image
--------------------------

|docker-size|

Get the |project| docker image by

.. prompt:: bash

  docker pull sameli/imate

The docker image has the followings pre-installed:

* CUDA: in ``/usr/local/cuda``
* Python 3.9: in ``/usr/bin/python3``
* Python interpreters: `ipython`, `jupyter`
* Editor: `vim`

Use |project| Docker Container on GPU
-------------------------------------
      
To use host's GPU from the docker container, add  ``--gpus all`` to any of the ``docker run`` commands, such as by

.. prompt:: bash

    docker run --gpus all -it sameli/imate

The followings are some examples of using ``docker run`` with various options:

* To check the host's NVIDIA driver version, CUDA runtime library version, and list of available GPU devices, run ``nvida-smi`` command by:

  .. prompt:: bash
  
      docker run --gpus all sameli/imate nvidia-smi
  
* To run the container and open *Python* interpreter directly at startup:
  
  .. prompt:: bash
  
      docker run -it --gpus all sameli/imate
  
  This also imports |project| package automatically.
  
* To run the container and open *IPython* interpreter directly at startup:
  
  .. prompt:: bash

        docker run -it --gpus all sameli/imate ipython
  
  This also imports `imate` package automatically.
  
* To open *Bash shell* only:
  
  .. prompt:: bash

        docker run -it --gpus all --entrypoint /bin/bash sameli/imate
  
* To *mount* a host's directory, such as ``/home/user/project``, onto a directory of the docker's container, such as ``/root``, use:
  
  .. prompt:: bash
  
        docker run -it --gpus all -v /home/user/project:/root sameli/imate

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

.. _gpu-cluster:

Deploy |project| on GPU Clusters
================================

On GPU clusters, the NVIDIA graphic driver and CUDA libraries are pre-installed and they only need to be loaded.

Load Modules
------------

Check which modules are available on the machine

.. prompt:: bash
    
    module avail

Load python and a compatible CUDA version by

.. prompt:: bash

    module load python/3.9
    module load cuda/11.7

Check which modules are loaded

.. prompt:: bash

    module list

Interactive Session with SLURM
------------------------------

There are two ways to work with GPU on a cluster. The first method is to ``ssh`` to a GPU node and for hands-on interaction with the GPU device. If the GPU cluster uses `SLURM manager <https://slurm.schedmd.com/documentation.html>`_, use ``srun`` to initiate a session as follows

.. prompt:: bash

    srun -A fc_biome -p savio2_gpu --gres=gpu:1 --ntasks 2 -t 2:00:00 --pty bash -i

In the above example:

* ``-A fc_biome`` sets the group account associated with the user.
* ``-p savio2_gpu`` sets the name of the GPU node.
* ``--gres=gpu:1`` requests one GPU device on the node.
* ``--ntasks 2`` requests two parallel CPU threads on the node.
* ``-t 2:00:00`` requests a two-hour session.
* ``--pty bash`` starts a Bash shell.
* ``-i`` redirects std input to the user's terminal for interactive use.

See the list of `options of srun <https://slurm.schedmd.com/srun.html>`_ for details. As another example, to request a GPU node named ``savio2_1080ti`` with 4 GPU devices and 8 CPU threads for 10 hours, run

.. prompt:: bash

    srun -A fc_biome -p savio2_1080ti --gres=gpu:4 --ntasks 8 -t 10:00:00 --pty bash -i

.. note::

    Replace the name of nodes and accounts in the above example with yours. The name of GPU nodes and accounts in the above examples are obtained from `SAVIO Cluster <https://docs-research-it.berkeley.edu/services/high-performance-computing/overview/>`_ (an institutional Cluster at UC Berkeley).

Submit Jobs to GPU with SLURM
-----------------------------

To submit a parallel job to GPU nodes on a cluster with `SLURM manager`, use ``sbatch`` command, such as

.. prompt:: bash

    sbatch jobfile.sh

See the list of `options of sbatch <https://slurm.schedmd.com/sbatch.html>`_ for details. A sample job file, ``jobfile.sh`` is shown below. The highlighted line in the file instructs `SLURM` to request the number of GPU devices with ``--gres`` option.

.. code-block:: Slurm
   :emphasize-lines: 11

    #!/bin/bash

    #SBATCH --job-name=your_project
    #SBATCH --mail-type=your_email
    #SBATCH --mail-user=your_email
    #SBATCH --partition=savio2_1080ti
    #SBATCH --account=fc_biome
    #SBATCH --qos=savio_normal
    #SBATCH --time=72:00:00
    #SBATCH --nodes=1
    #SBATCH --gres=gpu:4
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task=8
    #SBATCH --mem=64gb
    #SBATCH --output=output.log

    # Point to where Python is installed
    PYTHON_DIR=$HOME/programs/miniconda3

    # Point to where a script should run
    SCRIPTS_DIR=$(dirname $PWD)/scripts

    # Directory of log files
    LOG_DIR=$PWD

    # Load modules
    module load cuda/11.2

    # Export OpenMP variables
    export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

    # Run the script
    $PYTHON_DIR/bin/python ${SCRIPTS_DIR}/script.py > ${LOG_DIR}/output.txt

In the above job file, modify ``--partition``, ``--account``, and ``--qos`` according to your user account allowance on the cluster.

.. |repo-size| image:: https://img.shields.io/github/repo-size/ameli/imate
   :target: https://github.com/ameli/imate
.. |docker-size| image:: https://img.shields.io/docker/image-size/sameli/imate
   :target: https://hub.docker.com/r/sameli/imate
