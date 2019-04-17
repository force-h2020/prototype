py -3 -m pip install --upgrade pip
py -3 -m pip install cython
cd ..
cd navigator
py -3 setup_pyx.py build_ext --inplace
cd ..
cd deployment\setup_files
