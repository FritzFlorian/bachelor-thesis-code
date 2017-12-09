import os

EXEC_ENDING = ''
if os.name == 'Windows':
    EXEC_ENDING = '.exe'

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

AI_TRIVIAL_PATH = os.path.join(ROOT_DIR, 'bin/ai_trivial{}'.format(EXEC_ENDING))

SELFPLAY_NN_SERVER_PORT = 5100
SELFEVAL_NN_SERVER_PORT = 5101
