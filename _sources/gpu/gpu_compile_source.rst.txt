.. _gpu-compile-source:

Compile |project| from Source with CUDA
=======================================

.. contents::

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
                cuda-nvcc-12-2 \
                libcublas-12-2 \
                libcublas-dev-12-2 \
                libcusparse-12-2 -y \
                libcusparse-dev-12-2

    .. tab-item:: CentOS 7
        :sync: centos

        .. prompt:: bash

            sudo yum install --setopt=obsoletes=0 -y \
                cuda-nvcc-12-2.x86_64 \
                cuda-cudart-devel-12-2.x86_64 \
                libcublas-12-2.x86_64 \
                libcublas-devel-12-2.x86_64 \
                libcusparse-12-2.x86_64 \
                libcusparse-devel-12-2.x86_64

    .. tab-item:: RHEL 9
        :sync: rhel

        .. prompt:: bash

            sudo dnf install --setopt=obsoletes=0 -y \
                cuda-nvcc-12-2.x86_64 \
                cuda-cudart-devel-12-2.x86_64 \
                libcublas-12-2.x86_64 \
                libcublas-devel-12-2.x86_64 \
                libcusparse-12-2.x86_64 \
                libcusparse-devel-12-2.x86_64

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

.. |repo-size| image:: https://img.shields.io/github/repo-size/ameli/imate
   :target: https://github.com/ameli/imate
