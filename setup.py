from distutils.core import setup
from Cython.Build import cythonize

setup(
    name='bachelor_thesis_code',
    version='0.1',
    description='Code of my bachelor thesis',
    url='bachelor-thesis-code',
    author='Fritz Florian',
    license='MIT',
    packages=['hometrainer'],
    dependency_links=['git+https://github.com/FritzFlorian/hometrainer.git@master'],
    install_requires=[
        'Cython',
        'matplotlib',
        'numpy',
        'pyzmq',
        'tensorflow'
    ],
    zip_safe=False,
    ext_modules=cythonize(["reversi/**/*.pyx", "reversialphazero/**/*.pyx"])
)
