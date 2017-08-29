# Rodent sim

Work in progress.

## Build Rodent

    $ git clone https://github.com/miyosuda/rodent.git
    $ cd rodent
    $ cmake .
    $ make -j4

For python3,

    $ cmake . -DBUILD_PYTHON3=ON
    
TODO: Direct include/lib path setting for numpy on MacOSX

## Install

    $ python setup.py bdist_wheel
    $ cd dist
    $ pip install rodent-0.0.1-py2-none-any.whl
