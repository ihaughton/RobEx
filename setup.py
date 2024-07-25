#!/usr/bin/env python

from setuptools import find_packages
from setuptools import setup
import shlex
import subprocess


def git_version():
    cmd = 'git log --format="%h" -n 1'
    return subprocess.check_output(shlex.split(cmd)).decode()


version = git_version()


def get_install_requires():
    install_requires = []
    with open('requirements.txt') as f:
        for req in f:
            install_requires.append(req.strip())
    return install_requires


setup(
    name='RobEx',
    version=version,
    packages=find_packages(),
    install_requires=get_install_requires(),
    author='Iain Haughton',
    author_email='iain.haughton@gmail.com',
)
