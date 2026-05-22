from setuptools import setup, find_packages
from Cython.Build import cythonize
import numpy

setup(
    name="kitsune-ids",
    packages=find_packages(),
    ext_modules=cythonize([
        "AfterImage_extrapolate.pyx"
    ]),
    include_dirs=[numpy.get_include()],
    zip_safe=False,  # Required for Cython
)