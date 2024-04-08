# setup.py
from setuptools import find_packages, setup

setup(
    name='cancer_data_analysis',
    version='0.1',
    packages=find_packages('src'),
    package_dir={'': 'src'},
)
