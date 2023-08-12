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
