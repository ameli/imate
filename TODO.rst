Change UseSparse to Sparse
Change UseLanczosTridiagonalization to Symmetric

===============
Troubleshooting
===============

Windows, python 2.7

ImportError: DLL load failed: The application has failed to start because its side-by-side configuration is incorrect. Please see the application event log or use the command-line sxstrace.exe tool for more detail.

Open powershell (run as administrator)
choco install vcpython27 -f -y



Put seaborn in try catch so it it is not installed, the package still work.

before building the package: install numpy (conda install -c conda-forge numpy -y)
