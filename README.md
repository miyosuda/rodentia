![rodent_logo](./doc/image/rodent_logo.png)

*Rodent* is a 3D learning environment for MacOSX and Linux.

You can easily design learning tasks with python script. All of the scene objects can be updated with regid body simulation.

[![preview](./doc/image/preview.png)](https://youtu.be/6thMDZlAzkk)

## Getting started on MacOSX

### For Python3

    $ git clone https://github.com/miyosuda/rodent.git
    $ cd rodent
    $ cmake . -DBUILD_PYTHON3=ON
    $ make -j4
    $ python3 setup.py bdist_wheel
    $ pip3 install dist/rodent-0.1.3-py3-none-any.whl

And then run example

    $ python3 examples/01_seekavoid_arena/main.py

### For Python2

    $ git clone https://github.com/miyosuda/rodent.git
    $ cd rodent
    $ cmake .
    $ make -j4
    $ python setup.py bdist_wheel
    $ pip install dist/rodent-0.1.3-py2-none-any.whl

And then run example

    $ python examples/01_seekavoid_arena/main.py

If you have trouble under Homebrew environment, use

    $ cmake -DUSE_HOMEBREW=ON .

instead of `$ cmake .`

(When version of python exe and python library doesn't match, an error ocurrs like this,

    Fatal Python error: PyThreadState_Get: no current thread
    Abort trap: 6

Then try this `-DUSE_HOMEBREW=ON` option.)



## Getting started on Ubuntu


### For Python3

    $ sudo apt-get install -y python3-dev
    $ git clone https://github.com/miyosuda/rodent.git
    $ cd rodent
    $ cmake . -DBUILD_PYTHON3=ON
    $ make -j4
    $ python3 setup.py bdist_wheel
    $ pip3 install dist/rodent-0.1.3-py3-none-any.whl

And then run example

    $ python3 examples/01_seekavoid_arena/main.py

### For Python2

    $ sudo apt-get install -y python-dev
    $ git clone https://github.com/miyosuda/rodent.git
    $ cd rodent
    $ cmake .
    $ make -j4
    $ python setup.py bdist_wheel
    $ sudo pip install dist/rodent-0.1.3-py2-none-any.whl

And then run example

    $ python examples/01_seekavoid_arena/main.py


## API

[Python API](doc/python_api.md)
