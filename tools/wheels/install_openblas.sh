# SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the license found in the LICENSE.txt file in the root
# directory of this source tree.


# =====================
# What this script does
# =====================
#
# This script installs OpenBLAS development libraries. In addition, it creates
# pkgconfig file for OpenBLAS (in /usr/lib64/pkgconfig/openblas.pc), so that
# pkgconfig can find OpenBLAS. PkgConfig is a tool similar to cmake, which is
# used by meson to find packages during build.
#
# Installing OpenBLAS is only needed when the Python platform is PyPy, not
# CPython. The reason for this is that we only need OpenBLAS to compile
# numpy and scipy from source (from their tar.gz files not from their wheels).
# In CPython, the wheel for numpy and scipy is always available for all python
# versions. In contrastm for PyPy, the wheel for numpy and scipy is usually not
# available. Hence, meson tries to build them from source, which in turn, it
# needs OpenBLAS for building them. As such, we only need OpenBLAS in PyPY.
#
# =====
# USAGE
# =====
#
# bash ./tools/wheels/install_openblas.sh ${{ matrix.python_version }}
#
# where {{ $matrix.python_version }} is the ninja language variable in
# deploy-pypi.yaml files (github action files) and it is a string containing
# the python version. For Cpython, this string is something like "cp39" and for
# PyPy, it contains something like "pp39".


set -xe

# Determine whether the python is CPython Or PyPy (such as "cp39" or "pp39")
INPUT_PYTHON=$1

# Check the input Python contains the PyPy keyword
if [[ $INPUT_PYTHON == *"pp"* ]];
then

    echo "PyPy detected. Install OpenBLAS."

    UNAME_STR=$(uname)

    if [[ ${UNAME_STR} == "Linux" ]];
    then

        # Install OpenBLAS
        yum update
        yum install -y openblas-devel
        yum install -y pkgconfig

        # Get OpenBLAS version
        OPENBLAS_VER=`rpm -qi openblas-devel | grep "Version" | cut -d":" -f2`

        # Make pkgconfig to be aware of OpenBLAS
        CONTENT="prefix=/usr
        exec_prefix=\${prefix}
        libdir=\${exec_prefix}/lib64
        includedir=\${prefix}/include/openblas

        Name: OpenBLAS
        Description: OpenBLAS Library
        Version: ${OPENBLAS_VER}
        Libs: -L\${libdir} -lopenblas
        Cflags: -I\${includedir}
        "

        echo "$CONTENT" | tee /usr/lib64/pkgconfig/openblas.pc

    elif [[ ${UNAME_STR} == "Darwin" ]];
    then

        brew install openblas
        export PKG_CONFIG_PATH="/usr/local/opt/openblas/lib/pkgconfig"
        # export LDFLAGS="-L/usr/local/opt/openblas/lib"
        # export CPPFLAGS="-I/usr/local/opt/openblas/include"

    fi

else

    # No need to install OpenBLAS when CPython is used.
    echo "PyPy not detected. Possibly Python is CPython. Nothing to do."

fi
