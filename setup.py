#!/usr/bin/env

from setuptools import setup

setup(
  name = 'amp',
  version = '1.2',
  long_description = open('README.md').read(),
  packages = ['amp', 'descriptor', 'regression'],
  package_dir = {'amp': '.',
  'descriptor' : 'descriptor',
  'regression' : 'regression',},
  classifiers = [
      'Programming Language :: Python',
      'Programming Language :: Python :: 2.6',
      'Programming Language :: Python :: 2.7',
      'Programming Language :: Python :: 3',
      'Programming Language :: Python :: 3.3',
  ],
  install_requires=['numpy>=1.7.0', 'matplotlib', 'ase'])
