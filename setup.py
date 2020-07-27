#! /usr/bin/env python
#
# Copyright (C) 2020 Vinnie Monaco <contact@vmonaco.com>

import os, sys
from setuptools import setup, Extension

def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()

exec(compile(open('stick/version.py').read(),
                  'stick/version.py', 'exec'))

install_requires = [
    'numpy',
    'brian2',
    'matplotlib',
]

setup_requires = [
    'pytest-runner',
]

tests_require = [
    'pytest',
]

setup_options = dict(
    name='stick',
    version=__version__,
    author='Vinnie Monaco',
    author_email='contact@vmonaco.com',
    description='Spike time interval computational kernel',
    license='GNU GPLv3',
    keywords='spiking neural network',
    url='https://github.com/vmonaco/stick',
    packages=['stick'],
    long_description=read('README.txt'),
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'License :: OSI Approved',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'Topic :: Software Development',
        'Topic :: Scientific/Engineering',
        'Programming Language :: Cython',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    install_requires=install_requires,
    setup_requires=setup_requires,
    tests_require=tests_require,
    package_data={
        "stick": [
            "../README.md",
            "../README.txt",
            "../LICENSE",
            "../MANIFEST.in",
        ]
    },
)

if __name__ == '__main__':
    setup(**setup_options)
