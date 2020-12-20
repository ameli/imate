set -e
set -x

cd /io
/opt/python/cp37-cp37m/bin/python setup.py bdist_wheel
/opt/python/cp38-cp38/bin/python setup.py bdist_wheel
cd dist
ls *.whl | xargs -L1 auditwheel repair 
rm *.whl
mv wheelhouse/*.whl .
