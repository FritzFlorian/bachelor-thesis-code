# TODO: Properly compile this before running on OTHR computers
import pyximport; pyximport.install()

import reinforcement.distribution as distribution
from reversi.game_core import Board, GameState
import zmq
import threading
import logging
import time


def main():
    logging.basicConfig(level=logging.DEBUG)

    thread = threading.Thread(target=test_client)
    thread.start()

    selfplay_server = distribution.SelfplayServer('tcp://localhost:5101')
    selfplay_server.run()


def test_client():
    # Counterpart for testing
    context = zmq.Context()
    server = context.socket(zmq.REP)
    server.bind('tcp://*:5101')

    with open('simple_8_by_8.map') as file:
        board = Board(file.read())

    game_state = GameState(board)
    game_state.get_next_possible_moves()

    while True:
        req = server.recv_pyobj()
        print('recv {} evaluations...'.format(len(req.last_evaluations)))
        server.send_pyobj(distribution.SelfplayServer.WorkResponse(7, 'reinforcement.distributed_8_by_8.neural_network.SimpleNeuralNetwork', None, [board], 128))


if __name__ == '__main__':
    main()
