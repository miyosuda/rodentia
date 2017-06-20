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

## Build
    $ cd rodent
    $ cmake .
    $ make
    
TODO: Direct include/lib path setting for numpy on MacOSX