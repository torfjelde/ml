#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

requirements = ['Click>=6.0',
                'numpy>=1.14.0',
                'scikit-learn>=0.19.1',
                'scipy>=1.1.0',
                'matplotlib>=2.2.2',
                'tqdm>=4.23.4',]

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', ]

setup(
    author="Tor Erlend Fjelde",
    author_email='tor.erlend95@gmail.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    description="Machine learning package related to summer project 2018.",
    entry_points={
        'console_scripts': [
            'rbm=ml.cli:rbm'
        ],
    },
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='ml',
    name='ml',
    packages=find_packages(),
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/torfjelde/ml',
    version='0.1.0',
    zip_safe=False,
)
