# SPDX-FileCopyrightText: Copyright 2021, Siavash Ameli <sameli@berkeley.edu>
# SPDX-License-Identifier: BSD-3-Clause
# SPDX-FileType: SOURCE
#
# This program is free software: you can redistribute it and/or modify it
# under the terms of the license found in the LICENSE.txt file in the root
# directory of this source tree.


# =======
# Imports
# =======

import sys


# ===========
# is notebook
# ===========

def is_notebook():
    """
    Returns ``True`` if this script is running in a jupyter notebook. Returns
    ``False`` otherwise, including both ipython and python.
    """

    if "ipykernel" in sys.modules:
        # Any modern front-end built on ipykernel (Colab, Kaggle, VS Code)
        return True

    try:
        from IPython import get_ipython
    except ImportError:
        return False

    ip = get_ipython()

    if ip is None:
        return False

    else:
        shell = ip.__class__.__name__

        if shell == 'ZMQInteractiveShell':
            # Jupyter notebook or qtconsole
            return True

        elif shell == 'TerminalInteractiveShell':
            # Terminal running IPython
            return False

        elif 'google.colab' in str(type(shell)):
            # Colab's shell (older runtimes)
            return True

        else:
            # Other type
            return False


# ============
# script guard
# ============

if __name__ == "__main__":
    print(is_notebook())
