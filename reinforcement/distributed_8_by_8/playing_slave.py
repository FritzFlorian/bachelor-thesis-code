# TODO: Properly compile this before running on OTHR computers
import pyximport; pyximport.install()

import reinforcement.distribution as distribution
import definitions
import logging

logging.basicConfig(level=logging.DEBUG)


def main():
    selfplay_server = distribution.PlayingSlave('tcp://127.0.0.1:{}'.format(definitions.TRAINING_MASTER_PORT))
    selfplay_server.run()


if __name__ == '__main__':
    main()
