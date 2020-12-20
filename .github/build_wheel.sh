# Run this script at the root directory of the package (at "/", where "/.github" exists) by:
#
# docker run -i -t -v `pwd`:/io quay.io/pypa/manylinux1_x86_64 /bin/bash /io/.github/build_wheel.sh

set -e
set -x

# /io is mapped to the pwd, which is the root of the repository
cd /io

# Build wheels for various python
/opt/python/cp27-cp27m/bin/python setup.py bdist_wheel
/opt/python/cp35-cp35m/bin/python setup.py bdist_wheel
/opt/python/cp36-cp36m/bin/python setup.py bdist_wheel
/opt/python/cp37-cp37m/bin/python setup.py bdist_wheel
/opt/python/cp38-cp38/bin/python setup.py bdist_wheel
/opt/python/cp39-cp39/bin/python setup.py bdist_wheel

# Wheels are written to /dist/*.whl
cd dist

# Repair wheels one by one. Repaired wheels will be written to /dist/wheelhouse.
ls *.whl | xargs -L1 auditwheel repair 

# Remove original wheels (in dist/) and replace them with manylinux wheels (in /dist/wheelhouse)
rm *.whl
mv wheelhouse/*.whl .
rm -r wheelhouse

# Test
/opt/python/cp38-cp38/bin/python -m pip install *cp38-cp38-manylinux1_x86_64.whl

cd ../.github
/opt/python/cp38-cp38/bin/python ./slq.py
