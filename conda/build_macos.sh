set -e
set -x


# brew install libomp
export CFLAGS="${CFLAGS} -isysroot /Library/Developer/CommandLineTools/SDKs/MacOSX10.15.sdk"
export SDKROOT=/Library/Developer/CommandLineTools/SDKs/MacOSX10.15.sdk

{{ PYTHON }} setup.py install --single-version-externally-managed --record=record.txt -vvv

# - export CFLAGS="${CFLAGS} -isysroot ${CONDA_BUILD_SYSROOT}"; {{ PYTHON }} setup.py install --single-version-externally-managed --record=record.txt
# - python setup.py install
# - export LDFLAGS=-L/usr/local/opt/libomp/lib   # [osx]
