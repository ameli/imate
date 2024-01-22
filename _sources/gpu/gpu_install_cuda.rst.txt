.. _gpu-install-cuda:

Install NVIDIA CUDA Toolkit
===========================

The following instruction describes installing `CUDA` for `Ubuntu`, `CentOS`, and `Red Hat (RHEL)`. You may refer to `CUDA installation guide <https://developer.nvidia.com/cuda-downloads>`_ from NVIDIA developer documentation for other operating systems and platforms.

.. attention::

    NVIDIA does not support macOS. You can install the NVIDIA CUDA Toolkit on Linux and Windows only.

.. _install-cuda-runtime-lib:

Install CUDA Runtime Libraries
------------------------------

To download and install the CUDA Toolkit on both Linux and Windows, refer to the `NVIDIA Developer website <https://developer.nvidia.com/cuda-downloads>`__. It's important to note that NVIDIA's installation instructions on their website include the entire CUDA Toolkit, which is typically quite large (over 6 GB in size).

However, for running |project|, you don't need to install the entire CUDA Toolkit. Instead, only a few of the CUDA runtime libraries, are required. Below are simplified installation instructions for Linux, allowing you to perform a minimal CUDA installation with only the necessary libraries.

.. _add_cuda_runtime_repos:

Add CUDA Repository
~~~~~~~~~~~~~~~~~~~

Before installing CUDA libraries, add CUDA repository to your package manager:

.. tab-set::

    .. tab-item:: Ubuntu/Debian
        :sync: ubuntu

        .. prompt:: bash

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

    .. tab-item:: CentOS 7
        :sync: centos

        .. prompt:: bash

            # Machine architecture
            ARCH=$(uname -m | grep -q -e 'x86_64' && echo 'x86_64' || echo 'sbsa')

            # OS Version
            OS_VERSION=$(awk -F= '/^VERSION_ID/{gsub(/"/, "", $2); print $2}' /etc/os-release)

            # Add CUDA Repository 
            sudo yum install -y yum-utils
            sudo yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel${OS_VERSION}/${ARCH}/cuda-rhel${OS_VERSION}.repo

    .. tab-item:: RHEL 9
        :sync: rhel

        .. prompt:: bash

            # Machine architecture
            ARCH=$(uname -m | grep -q -e 'x86_64' && echo 'x86_64' || echo 'sbsa')

            # OS Version
            OS_VERSION=$(awk -F= '/^VERSION_ID/{gsub(/"/, "", $2); print $2}' /etc/os-release)

            # Add CUDA Repository 
            sudo dnf install -y dnf-utils
            sudo dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel${OS_VERSION}/${ARCH}/cuda-rhel${OS_VERSION}.repo

Install Minimal CUDA Libraries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

In the following, you may change ``CUDA_VERSION`` to the CUDA version that you wish to install.

.. tab-set::

    .. tab-item:: Ubuntu/Debian
        :sync: ubuntu

        .. prompt:: bash

            # Set to the desired cuda version
            CUDA_VERSION="12-3"

            # Install required CUDA libraries
            sudo apt-get update
            sudo apt install -y \
                cuda-cudart-${CUDA_VERSION} \
                libcublas-${CUDA_VERSION} \
                libcusparse-${CUDA_VERSION}

    .. tab-item:: CentOS 7
        :sync: centos

        .. prompt:: bash

            # Choose a desired cuda version
            CUDA_VERSION="12-3"

            # Install required CUDA libraries
            sudo yum install --setopt=obsoletes=0 -y \
                cuda-cudart-${CUDA_VERSION} \
                libcublas-${CUDA_VERSION} \
                libcusparse-${CUDA_VERSION}

    .. tab-item:: RHEL 9
        :sync: rhel

        .. prompt:: bash

            # Choose a desired cuda version
            CUDA_VERSION="12-3"

            # Install required CUDA libraries
            sudo dnf install --setopt=obsoletes=0 -y \
                cuda-nvcc-${CUDA_VERSION} \
                libcublas-${CUDA_VERSION} \
                libcusparse-${CUDA_VERSION}

Export ``LD_LIBRARY_PATH`` environment variable with the CUDA library location by

.. prompt:: bash

    echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64${PATH:+:${LD_LIBRARY_PATH}}' >> ~/.bashrc
    source ~/.bashrc

.. _install-graphic-driver:

Install NVIDIA Graphic Driver
-----------------------------

First, make sure you have :ref:`added CUDA repository <add_cuda_runtime_repos>`. Then, install *NVIDIA graphic driver* with

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
