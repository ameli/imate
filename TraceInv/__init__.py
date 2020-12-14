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

if (UserCurrentDirectory == ProjectDirectory) or (ExecutableDirectory == ProjectDirectory):

    # In such case, either the current directory or executable directory is on sys.path prior to the built package.
    # sys.path.insert(0,os.path.abspath(os.path.join(ProjectDirectory,'lib','TraceInv')))
    # sys.path.insert(0,os.path.abspath(os.path.join(ProjectDirectory,'build','lib.linux-x86_64-3.8')))
    # sys.path.insert(0,'/home/sia/work/ADCP/Noise-Estimation/code/TraceInv/build/lib.linux-x86_64-3.8')
    print('ADDED')
    # sys.path.remove(ProjectDirectory)

# sys.path.insert(0,os.path.abspath('../build/lib.linux-x86_64-3.8'))
# sys.path.insert(0,'/home/sia/work/ADCP/Noise-Estimation/code/TraceInv/build/lib.linux-x86_64-3.8')
# sys.path.insert(0,'/home/sia/work/ADCP/Noise-Estimation/code/TraceInv/build')
# sys.path.insert(0,'/opt/miniconda3/lib/python3.8/site-packages/TraceInv-0.0.8-py3.8-linux-x86_64.egg')

for i in sys.path:
    print(i)

# # The RecursiveGolb.py should be located in '/docs'.
# import RecursiveGlob
#
# # Build (assuming we build cython WITHOUT '--inplace', that is: 'python setup.py build_ext' only.
# BuildDirectory = os.path.join('..','build')
#
# # Regex for pattern of lib files. Note: this is OS dependant. macos: *.dylib. Windows: *.dll
# LibFilePatterns = ['*.so','*.dylib','*.dll']
#
# # Find list of subdirectories of the build directory that have files with the pattern
# BuildSubDirectories = RecursiveGlob.RecursiveGlob(BuildDirectory,LibFilePatterns)
#
# # Append the subdirectories to the path
# for BuildSubDirectory in BuildSubDirectories:
#
#     # Note: the subdirectory is *relative* to the BuildDirectory.
#     Path = os.path.join(BuildDirectory,BuildSubDirectory)
#     sys.path.insert(0,os.path.abspath(Path))


# Import sub-packages
from .ComputeLogDeterminant import ComputeLogDeterminant
from .ComputeTraceOfInverse import ComputeTraceOfInverse
from .InterpolateTraceOfInverse import InterpolateTraceOfInverse
from .GenerateMatrix import GenerateMatrix

# Load version
from.__version__ import __version__
