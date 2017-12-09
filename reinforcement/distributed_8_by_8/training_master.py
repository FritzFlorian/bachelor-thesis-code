# TODO: Properly compile this before running on OTHR computers
import pyximport; pyximport.install()

import reinforcement.distribution as distribution
from reversi.game_core import Board
import logging
import os


def main():
    logging.basicConfig(level=logging.DEBUG)

    work_dir = os.path.join(os.path.curdir, 'test')
    nn_name = 'reinforcement.distributed_8_by_8.neural_network.SimpleNeuralNetwork'

    with open('simple_8_by_8.map') as file:
        board = Board(file.read())

    training_master = distribution.TrainingMaster(work_dir, nn_name, [board])
    training_master.run()


if __name__ == '__main__':
    main()
