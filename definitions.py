import os

EXEC_ENDING = ''
if os.name == 'Windows':
    EXEC_ENDING = '.exe'

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
PYPY_EXEC = os.path.join(ROOT_DIR, 'pypy-venv/bin/python{}'.format(EXEC_ENDING))
CPYTHON_EXEC = os.path.join(ROOT_DIR, 'cpython-venv/bin/python{}'.format(EXEC_ENDING))
