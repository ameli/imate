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

To download and install the CUDA Toolkit on both Linux and Windows, refer to the `NVIDIA Developer website <https://developer.nvidia.com/cuda-downloads>`__. It's important to note that NVIDIA's installation instructions on their website include the entire CUDA Toolkit, which is typically quite large (over 6 GB in size).

However, for compiling |project|, you don't need to install the entire CUDA Toolkit. Instead, only the CUDA compiler and a few specific development libraries, such as cuBLAS and cuSparse, are required. Below are simplified installation instructions for Linux, allowing you to perform a minimal CUDA installation with only the necessary libraries. Note that in the following, you may change ``CUDA_VERSION`` to the CUDA version that you wish to install.

.. tab-set::

    .. tab-item:: Ubuntu/Debian
        :sync: ubuntu

        .. prompt:: bash

            # Set to the desired cuda version
            CUDA_VERSION="12-3"

            # Machine architecture
            ARCH=$(uname -m | grep -q -e 'x86_64' && echo 'x86_64' || echo 'sbsa')

            # OS Version
            UBUNTU_VERSION=$(awk -F= '/^VERSION_ID/{gsub(/"/, "", $2); print $2}' /etc/os-release)
            OS_VERSION=$(dpkg --compare-versions "$UBUNTU_VERSION" "ge" "22.04" && echo "2204" || echo "2004")

            # Add CUDA Repository 
            sudo apt update
            sudo apt install wget -y
            wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${OS_VERSION}/${ARCH}/cuda-keyring_1.1-1_all.deb -P /tmp
            sudo dpkg -i /tmp/cuda-keyring_1.1-1_all.deb
            rm /tmp/cuda-keyring_1.1-1_all.deb

            # Install required CUDA libraries
            sudo apt-get update
            sudo apt install -y \
                cuda-nvcc-${CUDA_VERSION} \
                libcublas-${CUDA_VERSION} \
                libcublas-dev-${CUDA_VERSION} \
                libcusparse-${CUDA_VERSION} \
                libcusparse-dev-${CUDA_VERSION}

    .. tab-item:: CentOS 7
        :sync: centos

        .. prompt:: bash

            # Set to the desired cuda version
            CUDA_VERSION="12-3"

            # Machine architecture
            ARCH=$(uname -m | grep -q -e 'x86_64' && echo 'x86_64' || echo 'sbsa')

            # OS Version
            OS_VERSION=$(awk -F= '/^VERSION_ID/{gsub(/"/, "", $2); print $2}' /etc/os-release)

            # Add CUDA Repository 
            sudo yum install -y yum-utils
            sudo yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel${OS_VERSION}/${ARCH}/cuda-rhel${OS_VERSION}.repo

            # Install required CUDA libraries
            sudo yum install --setopt=obsoletes=0 -y \
                cuda-nvcc-${CUDA_VERSION} \
                cuda-cudart-devel-${CUDA_VERSION} \
                libcublas-${CUDA_VERSION} \
                libcublas-devel-${CUDA_VERSION} \
                libcusparse-${CUDA_VERSION} \
                libcusparse-devel-${CUDA_VERSION}

    .. tab-item:: RHEL 9
        :sync: rhel

        .. prompt:: bash

            # Set to the desired cuda version
            CUDA_VERSION="12-3"

            # Machine architecture
            ARCH=$(uname -m | grep -q -e 'x86_64' && echo 'x86_64' || echo 'sbsa')

            # OS Version
            OS_VERSION=$(awk -F= '/^VERSION_ID/{gsub(/"/, "", $2); print $2}' /etc/os-release)

            # Add CUDA Repository 
            sudo dnf install -y dnf-utils
            sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel${OS_VERSION}/${ARCH}/cuda-rhel${OS_VERSION}.repo

            # Install required CUDA libraries
            sudo dnf install --setopt=obsoletes=0 -y \
                cuda-nvcc-${CUDA_VERSION} \
                cuda-cudart-devel-${CUDA_VERSION} \
                libcublas-${CUDA_VERSION} \
                libcublas-devel-${CUDA_VERSION} \
                libcusparse-${CUDA_VERSION} \
                libcusparse-devel-${CUDA_VERSION}

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
    python -m pip install .

.. |repo-size| image:: https://img.shields.io/github/repo-size/ameli/imate
   :target: https://github.com/ameli/imate
