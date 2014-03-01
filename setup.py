
import hrf_estimation
from distutils.core import setup
import setuptools

CLASSIFIERS = """\
Development Status :: 5 - Production/Stable
Intended Audience :: Science/Research
Intended Audience :: Developers
License :: OSI Approved
Programming Language :: Python
Programming Language :: Python :: 3
Topic :: Software Development
Operating System :: POSIX
Operating System :: Unix

"""

setup(
    name='hrf_estimation',
    description='A module for estimating Hemodynamical Response Function from functional MRI data',
    long_description=open('README.rst').read(),
    version=hrf_estimation.__version__,
    author='Fabian Pedregosa',
    author_email='fabian@fseoane.net',
    url='https://pypi.python.org/pypi/hrf_estimation',
    packages=['hrf_estimation'],
    classifiers=[_f for _f in CLASSIFIERS.split('\n') if _f],
    license='Simplified BSD'
)
