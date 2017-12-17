from distutils.core import setup
from Cython.Build import cythonize

setup(
    name="ReversiAI",
    ext_modules=cythonize(["reversi/**/*.pyx", "reinforcement/**/*.pyx"])
)