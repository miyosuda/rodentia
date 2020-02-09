from setuptools import setup

setup(
  name='rodent',

  version='0.1.3',

  description='3D reinforcement learning platform',
  long_description="Rodent is a 3D reinforcement learning platform.",

  url='https://github.com/miyosuda/rodent',

  author='Kosuke Miyoshi',
  author_email='miyosuda@gmail.com',

  install_requires=['numpy(>=1.11.0)'],
  packages=['rodent'],
  package_data={'rodent': ['rodent_module.so']},

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
  keywords=['rodent', 'ai', 'deep learning', 'reinforcement learning', 'research'],
)
