from setuptools import setup, find_packages

setup(
    name='FCSCPA',
    version='0.0.1',
    packages=find_packages('python'),
    package_dir={'':'python'},
    description='Full Counting Statistics using Coherent Potential Approximation',
    install_requires=['scipy','quadpy>=0.16.1'],
)
