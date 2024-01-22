#! /bin/bash

# SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the license found in the LICENSE.txt file in the root directory
# of this source tree.


# =============================================================================
# 
# This script installs minimal CUDA libraries (dev or runtime) required to
# compile the package or to run the package.
#
# Usage:
#
# sudo sh ./install cuda <CUDA_VERSION> <LIB_TYPE>
#
# In the above:
#
# <CUDA_VERSION> should be in the format of "xx-x", such as "12-3"
# <LIB_TYPE> can be "dev" or "rt".
#
# - "dev": This installs development libraries and NVCC compiler. This should
#          be used for compiling with CUDA.
# - "rt":  This installs CUDA runtime libraries and NVIDIA graphic drivers.
#          This should be used at runtime.
#
# =============================================================================


set -e
set -x

if [[ "$#" -lt 2 ]]; then
    echo "Usage: sudo $0 <CUDA_VERSION> [dev, rt]" >&2
    exit 1
fi

# First argument determines the CUDA version to be installed
CUDA_VERSION=$1
if [[ ${CUDA_VERSION} =~ "^[0-9]{2}-[0-9]$" ]];
then
    echo 'ERROR: First argument should be the cuda version with the format' \
        ' "xx-x".' >&2
    exit 1
fi

# Second argument determines what to be installed. "rt" means runtime libraries
# and "dev" means development liberties and NVCC compiler.
LIB_TYPE=$2
if [[ ! ${LIB_TYPE} == "dev" ]] && [[ ! ${LIB_TYPE} == "rt" ]];
then
    echo 'ERROR: Second argument should be either "dev" or "rt".' >&2
    exit 1
fi

# Name of OS (ubuntu, centos, rhel)
if [[ ! -e /etc/os-release ]];
then
    echo 'ERROR: File /etc/os-release does not exists.' >&2
    exit 1
fi

# Machine architecture
ARCH=$(uname -m | grep -q -e 'x86_64' && echo 'x86_64' || echo 'sbsa')

# Linux distro
DISTRO=$(awk -F= '/^ID=/{gsub(/"/, "", $2); print $2}' /etc/os-release)

# Get OS version depending on distro
case "$DISTRO" in

    # ------
    # Ubuntu
    # ------

    "ubuntu")

        # Get OS Version
        UBUNTU_VERSION=$(awk -F= '/^VERSION_ID/{gsub(/"/, "", $2); print $2}' /etc/os-release)
        OS_VERSION=$(dpkg --compare-versions "$UBUNTU_VERSION" "ge" "22.04" && echo "2204" || echo "2004")

        # Add CUDA repository
        apt update
        apt install wget -y
        wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu${OS_VERSION}/${ARCH}/cuda-keyring_1.1-1_all.deb -P /tmp
        dpkg -i /tmp/cuda-keyring_1.1-1_all.deb
        rm /tmp/cuda-keyring_1.1-1_all.deb

        # Install required CUDA libraries
        apt-get update

        case "$LIB_TYPE" in

            "dev")
                # Install developement libraries and compiler
                apt install -y \
                    cuda-nvcc-${CUDA_VERSION} \
                    libcublas-${CUDA_VERSION} \
                    libcublas-dev-${CUDA_VERSION} \
                    libcusparse-${CUDA_VERSION} \
                    libcusparse-dev-${CUDA_VERSION}
                ;;

            "rt")
                # Install runtime libraries and NVIDIA graphic driver
                export DEBIAN_FRONTEND=noninteractive
                apt install -y \
                    cuda-cudart-${CUDA_VERSION} \
                    libcublas-${CUDA_VERSION} \
                    libcusparse-${CUDA_VERSION} \
                    cuda-drivers
                ;;
        esac
        ;;

    # ------
    # CentOS
    # ------

    "centos")

        # Get OS Version
        OS_VERSION=$(awk -F= '/^VERSION_ID/{gsub(/"/, "", $2); print $2}' /etc/os-release)

        # Add CUDA repository
        yum install -y yum-utils
        yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel${OS_VERSION}/${ARCH}/cuda-rhel${OS_VERSION}.repo

        # Install required CUDA libraries
        case "$LIB_TYPE" in

            "dev")
                # Install developement libraries and compiler
                yum install --setopt=obsoletes=0 -y \
                    cuda-nvcc-${CUDA_VERSION} \
                    cuda-cudart-devel-${CUDA_VERSION} \
                    libcublas-${CUDA_VERSION} \
                    libcublas-devel-${CUDA_VERSION} \
                    libcusparse-${CUDA_VERSION} \
                    libcusparse-devel-${CUDA_VERSION}
                ;;

            "rt")
                # Install runtime libraries and NVIDIA graphic driver
                yum install --setopt=obsoletes=0 -y \
                    cuda-cudart-${CUDA_VERSION} \
                    libcublas-${CUDA_VERSION} \
                    libcusparse-${CUDA_VERSION} \
                    nvidia-driver-latest-dkms
                ;;

            esac
        ;;

    # ------
    # RedHat
    # ------

    "rhel")

        # Get OS Version
        OS_VERSION=$(awk -F= '/^VERSION_ID/{gsub(/"/, "", $2); print $2}' /etc/os-release)

        # Add CUDA repository
        dnf install -y dnf-utils
        dnf config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel${OS_VERSION}/${ARCH}/cuda-rhel${OS_VERSION}.repo

        # Install required CUDA libraries
        case "$LIB_TYPE" in

            "dev")
                # Install developement libraries and compiler
                dnf install --setopt=obsoletes=0 -y \
                    cuda-nvcc-${CUDA_VERSION} \
                    cuda-cudart-devel-${CUDA_VERSION} \
                    libcublas-${CUDA_VERSION} \
                    libcublas-devel-${CUDA_VERSION} \
                    libcusparse-${CUDA_VERSION} \
                    libcusparse-devel-${CUDA_VERSION}
                    ;;

            "rt")
                # Install runtime libraries and NVIDIA graphic driver
                dnf install --setopt=obsoletes=0 -y \
                    cuda-nvcc-${CUDA_VERSION} \
                    libcublas-${CUDA_VERSION} \
                    libcusparse-${CUDA_VERSION} \
                    nvidia-driver:latest-dkms
                ;;
        esac
        ;;

    *)

        echo 'Error: Invalid distribution. Please choose one of "ubuntu", "centos", or "rhel".'
        exit 1
        ;;

esac

# Add CUDA libraries to ld seach path
echo "/usr/local/cuda/lib64" >> /etc/ld.so.conf.d/999_nvidia_cuda.conf

# Add environment variables to bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/lib64${PATH:+:${LD_LIBRARY_PATH}}' >> ${HOME}/.bashrc
echo 'export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}' >> ${HOME}/.bashrc
