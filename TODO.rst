Change UseSparse to Sparse
Change UseLanczosTridiagonalization to Symmetric

===============
Troubleshooting
===============

Windows, python 2.7

ImportError: DLL load failed: The application has failed to start because its
side-by-side configuration is incorrect. Please see the application event log
or use the command-line sxstrace.exe tool for more detail.

Open powershell (run as administrator)
choco install vcpython27 -f -y



Put seaborn in try catch so it it is not installed, the package still work.

Before building the package: install numpy (conda install -c conda-forge numpy -y)

Before building package in windows
choco install -y visualstudio2019buildtools
choco install -y vcbuildtools
Or download Microsoft C++ Build Tools.

====
Name
====

Implicit Matrix Trace Estimator: imte, >>"imate"<<, imtraes, "imtrest",
    "tracest", "imtest"
Fast Trace Estimator
"scikit-trace"

=====
Ideas
=====

Encapsulate functions in a cdef class so that they can be passed from python to
slq method.


