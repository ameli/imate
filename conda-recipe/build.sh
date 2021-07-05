#! /bin/bash

set -e
set -x

# Get python version
python_version_major=`python -c 'import sys; print(sys.version_info.major)'`
python_version_minor=`python -c 'import sys; print(sys.version_info.minor)'`
python_version="${python_version_major}${python_version_minor}"

# Detect operating system
os=`uname`;
case "$os" in
    *"Linux"*) platform="linux";;
    *"Darwin"*) platform="macosx";;
    *) echo "Invalid operating system. os: $os."; exit 1;;
esac

# Iterate through all wheel files in /dist/* directory.
wheel_installed=false;
for wheel_filename in dist/*.whl;
do
    # Check if wheel filename matches python version and platform
    if echo $wheel_filename | grep $python_version | grep $platform; then

        # Matched. Install with pip
        echo "Try installing wheel file '${wheel_filename}'.";
        python -m pip install --upgrade pip
        python -m pip install --no-deps --force-reinstall $wheel_filename;

        # Check last error code
        error_code=$?;
        if [ "${error_code}" -eq 0 ];
        then
            echo "Wheel '${wheel_filename}' installed successfully.";
            wheel_installed=true;
            break;
        fi
    else
        # Did not match. Skip installing this wheel file.
        echo -n "Wheel file '${wheel_filename}' didn't match python version: ";
        echo "'${python_version}' and platform: '${platform}'. Skipping.";
    fi
done

if [ "$wheel_installed" != true ];
then
    echo "ERROR: No wheel could be installed." 1>&2;
    exit 1;
fi
