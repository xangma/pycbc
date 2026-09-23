from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

names = ['chisq_base', 'chisq_malloc', 'chisq_malloc1', 'chisq_cache', 'chisq_numpy']
compile_args = ['-O3', '-w', '-ffast-math', '-ffinite-math-only', '-msse4.2', '-fopenmp']
link_args = ['-fopenmp']
extensions = [Extension(name, [name + '.pyx'], language='c++',
                        include_dirs=[numpy.get_include()], libraries=['m'],
                        extra_compile_args=compile_args,
                        extra_link_args=link_args) for name in names]
setup(name='chisq-pr5452-benchmark', ext_modules=cythonize(extensions, quiet=True))
