set -e
set -x


# brew install libomp
# export CFLAGS="${CFLAGS} -isysroot=/Library/Developer/CommandLineTools/SDKs/MacOSX10.15.sdk -rpath ${PREFIX}/lib -L${PREFIX}/lib"
export CFLAGS="-isysroot=/Library/Developer/CommandLineTools/SDKs/MacOSX10.15.sdk -rpath ${PREFIX}/lib -L${PREFIX}/lib"
export SDKROOT="/Library/Developer/CommandLineTools/SDKs/MacOSX10.15.sdk"
echo "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA"
echo $CONDA_BUILD_SYSROOT
export CONDA_BUILD_SYSROOT="/Library/Developer/CommandLineTools/SDKs/MacOSX10.15.sdk"
echo $CONDA_BUILD_SYSROOT
echo "BBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBBB"

{{ PYTHON }} setup.py install --single-version-externally-managed --record=record.txt -vvv

# - export CFLAGS="${CFLAGS} -isysroot ${CONDA_BUILD_SYSROOT}"; {{ PYTHON }} setup.py install --single-version-externally-managed --record=record.txt
# - python setup.py install
# - export LDFLAGS=-L/usr/local/opt/libomp/lib   # [osx]
