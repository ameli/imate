#! /bin/bash

set -x

wheel_installed=false;
echo $wheel_installed;
for filename in dist/*.whl;
do
    if [ "$wheel_installed" != true ];
    then
        echo "Try installing ${filename} ...";
        python -m pip install --no-deps --force-reinstall $filename 2> /dev/null;
        error_code=$?;

        if [ "${error_code}" -eq 0 ];
        then
            wheel_installed=true;
            echo "Wheel ${filename} installed successfully.";
        fi
    fi
done

if [ "$wheel_installed" != true ];
then
    echo "ERROR: No wheel could be installed." 1>&2;
    exit 1;
fi
