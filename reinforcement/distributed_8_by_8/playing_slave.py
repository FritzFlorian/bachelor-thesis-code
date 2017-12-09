# TODO: Properly compile this before running on OTHR computers
import pyximport; pyximport.install()

import reinforcement.distribution as distribution
import definitions
import logging


def main():
    logging.basicConfig(level=logging.DEBUG)

    selfplay_server = distribution.PlayingSlave('tcp://localhost:{}'.format(definitions.TRAINING_MASTER_PORT))
    selfplay_server.run()


if __name__ == '__main__':
    main()
