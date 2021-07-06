#! /bin/bash

# SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it under
# the terms of the license found in the LICENSE.txt file in the root directory
# of this source tree.


set -e
set -x

# Choose an existing version from the list of CUDA packages here:
# https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/

VERSION=11-4
ARCH=x86_64

yum install -y yum-utils

yum-config-manager --add-repo https://developer.download.nvidia.com/compute/cuda/repos/rhel7/x86_64/cuda-rhel7.repo

yum install --setopt=obsoletes=0 -y \
    cuda-nvcc-$VERSION.$ARCH \
    cuda-cudart-devel-$VERSION.$ARCH \
    libcublas-$VERSION.$ARCH \
    libcublas-devel-$VERSION.$ARCH \
    libcusparse-$VERSION.$ARCH \
    libcusparse-devel-$VERSION.$ARCH

pip3 install --upgrade pip
ln -s cuda-$VERSION /usr/local/cuda
