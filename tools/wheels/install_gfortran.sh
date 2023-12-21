# SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the license found in the LICENSE.txt file in the root
# directory of this source tree.


set -xe
PROJECT_DIR="$1"
PLATFORM="macosx-arm64"
PLAT="arm64"
source $PROJECT_DIR/tools/wheels/gfortran_utils.sh
install_gfortran
pip install "delocate==0.10.4"
