from setuptools import setup

setup(
  name='rodentia',

  version='0.1.4',

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
    'Programming Language :: Python :: 3.3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
  ],

  # What does your project relate to?
  keywords=['rodentia', 'ai', 'deep learning', 'reinforcement learning', 'research'],
)
