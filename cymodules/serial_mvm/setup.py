from setuptools import Extension, setup
from Cython.Build import cythonize

ext_modules = [
    Extension(
        "mvm",
        ["mvm.pyx"],
    )
]

setup(
    name='mvm',
    ext_modules=cythonize(ext_modules),
)