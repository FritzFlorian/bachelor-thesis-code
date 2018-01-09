import os

# Settings for training/internal tournaments
EXEC_ENDING = ''
if os.name == 'Windows':
    EXEC_ENDING = '.exe'

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

AI_TRIVIAL_PATH = os.path.join(ROOT_DIR, 'bin/ai_trivial{}'.format(EXEC_ENDING))

SELFPLAY_NN_SERVER_PORT = 5100
SELFEVAL_NN_SERVER_PORT = 5101
TRAINING_NN_SERVER_PORT = 5102

TRAINING_MASTER_PORT = 5200

# Settings for the game core
REVERSI_MATCH_SERVER_DEFAULT_PORT = 7777
REVERSI_MATCH_SERVER_DEFAULT_HOST = 'localhost'
