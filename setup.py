from setuptools import setup

from vmitools import __version__


setup(
    name='vmitools',
    version=__version__,
    author='Daehyun You',
    author_email='daehyun@dc.tohoku.ac.jp',
    url='https://github.com/DaehyunPY/vmitools',
    # description='',
    # long_description='',
    license='MIT',
    packages=[
        'vmitools',
    ],
    install_requires=[
        'numpy',
        'numba',
        'cytoolz',
        'pandas',
        'sympy',
        'h5py',
    ],
)
