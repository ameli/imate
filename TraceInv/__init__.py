import sys
import os

# Find the current directory of the user (this is where the user calls an executable)
UserCurrentDirectory = os.getcwd()

# Find executable directory (this is where the *.py executable is)
ExecutableFile = os.path.abspath(sys.argv[0])
ExecutableDirectory = os.path.dirname(ExecutableFile)

# Find the project directory (this is the second parent directory of this file, __init__.py)
PackageDirectory = os.path.dirname(os.path.realpath(__file__))   # This means: ../
ProjectDirectory = os.path.dirname(PackageDirectory)             # This means: ../../

# Note: the following two issues (see if conditions below) only happen in a cython package and if the package is build without --inplace option.
# This is because without the --inplace option, the *.so files will be built inside the /build directory (not in the same directory of the source
# code where *.pyx files are). On the other hand, when the user's directory is in the parent directory of the package, this path will be
# the first path on the sys.path. Thus, it looks for the package in the source-code directory, not where it was built or installed. But
# because the built is outside of the source (recall no --inplace), it cannot find the *.so files.
#
# To resolve this issue:
# 1. Either build the package with --inplace option.
# 2. Change the current directory, or the directory of the script that you are running out of the source code.
if (UserCurrentDirectory == ProjectDirectory):
    print('You are in the source-code directory of the package. Importing the package will probably fail.:')
    print('To resolve the issue, consider changing the current directory outside of the parent directory of the source-code.')
    print('Your current directory is: %s'%UserCurrentDirectory)

if (ExecutableDirectory == ProjectDirectory):

    print('You are running a script in the source-code directory of the package. Importing the package will probably fail.')
    print('To resolve the issue, consider changing the executable directory.')
    print('The executable directory is: %s'%ExecutableDirectory)

# Import sub-packages
from .ComputeLogDeterminant import ComputeLogDeterminant
from .ComputeTraceOfInverse import ComputeTraceOfInverse
from .InterpolateTraceOfInverse import InterpolateTraceOfInverse
from .GenerateMatrix import GenerateMatrix

# Load version
from.__version__ import __version__
