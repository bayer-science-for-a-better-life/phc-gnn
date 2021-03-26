from setuptools import setup

setup(
    name='phc',
    version='0.1',
    packages=['phc', 'phc.hypercomplex', 'phc.quaternion', 'benchmarks'],
    url='https://github.com/bayer-science-for-a-better-life/phc-gnn',
    license='GPL-3',
    author='Tuan Le',
    author_email='tuan.le2@bayer.com',
    description='Implementation of the paper "Parameterized Hypercomplex Graph Neural Networks for Graph Classification.'
)