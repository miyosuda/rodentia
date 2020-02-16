from setuptools import setup, Extension

import os
from pathlib import Path
import subprocess
from setuptools.command.build_ext import build_ext

NAME = 'rodentia'
VERSION = None

about = {}

if not VERSION:
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, NAME, '__version__.py')) as f:
        exec(f.read(), about)
else:
    about['__version__'] = VERSION


class CMakeExtension(Extension):
    def __init__(self, name):
        Extension.__init__(self, name, sources=[])



class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError(
                "CMake must be installed to build the following extensions: " +
                ", ".join(e.name for e in self.extensions))
        
        build_directory = os.path.abspath(self.build_temp)

        cmake_args = [
            '-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + build_directory,
        ]
        
        build_args = []
        cmake_args += []

        # Assuming Makefiles
        build_args += ['--', '-j4']
        self.build_args = build_args
        
        env = os.environ.copy()
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)

        # CMakeLists.txt is in the same directory as this setup.py file
        cmake_list_dir = os.path.abspath(os.path.dirname(__file__))
        print('-' * 10, 'Running CMake prepare', '-' * 40)
        
        subprocess.check_call(['cmake', cmake_list_dir] + cmake_args,
                              cwd=self.build_temp,
                              env=env)

        print('-' * 10, 'Building extensions', '-' * 40)
        
        cmake_cmd = ['cmake', '--build', '.'] + self.build_args
        subprocess.check_call(cmake_cmd,
                              cwd=self.build_temp)

        for ext in self.extensions:
            self.copy_ext(ext.name)
            
    def copy_ext(self, ext_base_path):
        # Move from build temp to final position
        ext_base_name = os.path.basename(ext_base_path)
        build_temp = Path(self.build_temp).resolve()
        ext_local_path = ext_base_path + ".so"
        dst_dir = os.path.abspath(os.path.dirname(
            self.get_ext_fullpath(ext_base_path)))
        dst_path = Path(dst_dir).resolve() / (ext_base_name + ".so")
        src_path = build_temp / ext_local_path
        dst_dir_path = dst_path.parents[0]
        dst_dir_path.mkdir(parents=True, exist_ok=True)
        self.copy_file(str(src_path.absolute()), str(dst_path.absolute()))
        
setup(
    name=NAME,
    version=about['__version__'],

    description='3D reinforcement learning platform',
    long_description="Rodentia is a 3D reinforcement learning platform.",

    url='https://github.com/miyosuda/rodentia',

    author='Kosuke Miyoshi',
    author_email='miyosuda@gmail.com',

    install_requires=['numpy(>=1.11.0)'],
    packages=['rodentia'],
    package_data={'rodentia': ['rodentia_module.so']},

    license='Apache 2.0',

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: Apache Software License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],

    # What does your project relate to?
    keywords=['rodentia', 'ai', 'deep learning', 'reinforcement learning', 'research'],

    ext_modules=[
        CMakeExtension('rodentia/rodentia_module'),
    ],

    cmdclass={
        'build_ext' : CMakeBuild
    },

    zip_safe=False,
)
