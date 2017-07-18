# Always prefer setuptools over distutils
from setuptools import setup, find_packages

setup(
  name='rodent',

  version='0.0.1',

  description='3D reinforcement learning platform',
  long_description="Rodent is a 3D reinforcement learning platform.",

  url='https://github.com/miyosuda/rodent',

  author='Kosuke Miyoshi',
  author_email='miyosuda@gmail.com',

  install_requires=['numpy'],  
  packages=['rodent'],  
  package_data={'rodent': ['rodent', 'rodent_module.so']},

  license='Apache 2.0',

  classifiers=[
    'Development Status :: 2 - Pre-Alpha',

    'Intended Audience :: Science/Research',
    'Topic :: Scientific/Engineering :: Artificial Intelligence',

    'License :: OSI Approved :: Apache License, 2.0',

    # Specify the Python versions you support here. In particular, ensure
    # that you indicate whether you support Python 2, Python 3 or both.
    'Programming Language :: Python :: 2',
    'Programming Language :: Python :: 2.7',
    #'Programming Language :: Python :: 3',
    #'Programming Language :: Python :: 3.3',
    #'Programming Language :: Python :: 3.4',
    #'Programming Language :: Python :: 3.5',
  ],

  # What does your project relate to?
  keywords=['rodent', 'ai', 'deep learning', 'reinforcement learning', 'research'],
)
