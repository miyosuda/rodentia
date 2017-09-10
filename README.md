# Rodent

*Rodent* is a 3D learning environment for MacOSX and Linux.

## Getting started on MacOSX

For Python2

    $ git clone https://github.com/miyosuda/rodent.git
    $ cd rodent
    $ cmake .
	$ make -j4
    $ python setup.py bdist_wheel
    $ pip install dist/rodent-0.0.1-py2-none-any.whl

For Python3

    $ git clone https://github.com/miyosuda/rodent.git
    $ cd rodent
	$ cmake . -DBUILD_PYTHON3=ON
	$ make -j4
    $ python setup.py bdist_wheel
    $ pip install dist/rodent-0.0.1-py3-none-any.whl
