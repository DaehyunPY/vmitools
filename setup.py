from setuptools import setup


setup(
    name='vmitools',
    version='v201812.1',
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
