![rodentia_logo](./doc/image/rodentia_logo.png)

*Rodentia* is a 3D learning environment for MacOSX and Linux.

You can easily design learning tasks with python script. All of the scene objects can be updated with regid body simulation.

[![preview](./doc/image/preview.png)](https://youtu.be/6thMDZlAzkk)


## Install with pip

### MacOSX

First confirm `cmake` in installed.

    $ cmake --version

If `cmake` is not installed, install it.

    $ brew install cmake

And install with `rodentia` with pip

    $ pip3 install rodentia

### Ubuntu

First confirm `cmake` in installed.

    $ cmake --version

If `cmake` is not installed, install it.

    $ sudo apt-get install cmake


And install with `rodentia` with pip

    $ sudo apt-get install -y python3-dev
    $ pip3 install rodentia


## Build from source

### MacOSX

First confirm `cmake` in installed.

    $ cmake --version

If `cmake` is not installed, install it.

    $ brew install cmake

Then install `rodentia`

    $ git clone https://github.com/miyosuda/rodentia.git
    $ cd rodentia
    $ pip3 install .


### Ubuntu

First confirm `cmake` in installed.

    $ cmake --version

If `cmake` is not installed, install it.

    $ sudo apt-get install cmake


And install with `rodentia` with pip

    $ sudo apt-get install -y python3-dev
    $ git clone https://github.com/miyosuda/rodentia.git
    $ cd rodentia
    $ pip3 install .


## How to run example

    $ pip3 install pygame==2.0.0.dev6 Pillow>=5.1.0
    $ python3 examples/01_seekavoid_arena/main.py


## API

[Python API](doc/python_api.md)
