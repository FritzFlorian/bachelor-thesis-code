import os
import platform

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))

# Settings for training/internal tournaments
AI_TRIVIAL_PATH_UNIX = 'bin/ai_trivial'
AI_TRIVIAL_PATH_WINDOWS = 'bin/ai_trivial.exe'
AI_TRIVIAL_PATH_MAC = 'bin/ai_trivial_mac'
if platform.system() == 'Windows':
    AI_TRIVIAL_PATH = os.path.join(ROOT_DIR, AI_TRIVIAL_PATH_WINDOWS)
elif platform.system() == 'Darwin':
    AI_TRIVIAL_PATH = os.path.join(ROOT_DIR, AI_TRIVIAL_PATH_MAC)
else:
    AI_TRIVIAL_PATH = os.path.join(ROOT_DIR, AI_TRIVIAL_PATH_UNIX)

AI_JAVA_PATH = os.path.join(ROOT_DIR, 'bin/client.jar')

AI_TRIVIAL_AVAILABLE = os.path.isfile(AI_TRIVIAL_PATH)


# NN Server Settings
SELFPLAY_NN_SERVER_PORT = 5100
SELFEVAL_NN_SERVER_PORT = 5101
TRAINING_NN_SERVER_PORT = 5102

# Distribution settings
TRAINING_MASTER_PORT = 5200


WEB_INTERFACE_PORT = 5300
LOGGING_SERVER_PORT = 5301

# Settings for the game core
REVERSI_MATCH_SERVER_DEFAULT_PORT = 7777
REVERSI_MATCH_SERVER_DEFAULT_HOST = 'localhost'


# Settings for ZeroMQ security
KEYS_DIR = os.path.join(ROOT_DIR, 'keys')
PUBLIC_KEYS_DIR = os.path.join(KEYS_DIR, 'public_keys')
PRIVATE_KEYS_DIR = os.path.join(KEYS_DIR, 'private_keys')
SERVER_SECRET = os.path.join(PRIVATE_KEYS_DIR, 'server.key_secret')
SERVER_PUBLIC = os.path.join(PUBLIC_KEYS_DIR, 'server.key')
CLIENT_SECRET = os.path.join(PRIVATE_KEYS_DIR, 'client.key_secret')
CLIENT_PUBLIC = os.path.join(PUBLIC_KEYS_DIR, 'client.key')

