# Rodent sim

Work in progress.

## Preparation (for MacOSX)

### Install Bullet

	$ wget https://github.com/bulletphysics/bullet3/archive/2.86.1.tar.gz -O bullet3-2.86.1.tar.gz
	$ tar xvf bullet3-2.86.1.tar.gz
	$ cd bullet3-2.86.1
    $ cmake .
    $ make -j4
    $ make install

### Install Google Test

    $ git clone https://github.com/google/googletest.git
	$ cd googletest
    $ cmake .
    $ make -j4
    $ make install

### Install GLFW

    $ brew install glfw3

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
