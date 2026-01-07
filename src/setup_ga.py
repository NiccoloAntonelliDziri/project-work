from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "src.ga_solver",
        ["src/cython/ga_solver.pyx"],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
        include_dirs=[numpy.get_include()]
    ),
]

setup(
    name='ga_solver',
    ext_modules=cythonize(extensions, language_level=3),
)
