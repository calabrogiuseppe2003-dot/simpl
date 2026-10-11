# -*- coding: utf-8 -*-
try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

import sys
import os
import re
import glob


version = re.findall('__version__ = "(.*)"',
                     open('simpl/__init__.py', 'r').read())[0]

packages = [
    "simpl",
    ]

CLASSIFIERS = """
Development Status :: 2 - Pre-Alpha
Environment :: Console
Intended Audience :: Science/Research
License :: OSI Approved :: GNU Lesser General Public License v3 or later (LGPLv3+)
Programming Language :: Python
Topic :: Scientific/Engineering :: Mathematics
"""
classifiers = CLASSIFIERS.split('\n')[1:-1]

# TODO: This is cumbersome and prone to omit something
demofiles = glob.glob(os.path.join("examples", "*", "*.py"))
demofiles += glob.glob(os.path.join("examples", "*", "*", "*.py"))
demofiles += glob.glob(os.path.join("examples", "*", "*", "*.xml*"))
demofiles += glob.glob(os.path.join("examples", "*", "*", "*", "*.geo"))
demofiles += glob.glob(os.path.join("examples", "*", "*", "*", "*.xml*"))

# Don't bother user with test files
[demofiles.remove(f) for f in demofiles if "test_" in f]

setup(name="simpl",
      version=version,
      author="Guiseppe Calabrò and Ioannis Papadopoulos",
      url="https://github.com/calabrogiuseppe2003-dot/simpl",
      description="SiMPL",
      long_description="--",
      classifiers=classifiers,
      license="MIT Licence",
      packages=packages,
      package_dir={"simpl": "simpl"},
      data_files=[(os.path.join("share", "simpl", os.path.dirname(f)), [f])
                  for f in demofiles],
    )