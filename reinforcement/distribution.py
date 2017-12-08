import reinforcement.core as core
import reinforcement.nn_client as nn_client
from reversi.game_core import GameState
import multiprocessing
import zmq
import pickle
import logging
import definitions
import time
import numpy as np


class SelfplayServer:
    """Runs selfplay games using a given neural network configuration.

    The selfplay server reports it's results back to a master node.
    The master node coordinates workloads, stores results and configures the worker."""
    TIMEOUT = 30

    class WorkRequest:
        def __init__(self, last_evaluations):
            self.last_evaluations = last_evaluations

    class WorkResponse:
        def __init__(self, n_games, nn_name, weights_zip_binary, board_states, simulations_per_turn):
            self.n_games = n_games
            self.nn_name = nn_name
            self.weights_zip_binary = weights_zip_binary
            self.board_states = board_states
            self.simulations_per_turn = simulations_per_turn

    class WaitResponse:
        def __init__(self, wait_time):
            self.wait_time = wait_time

    def __init__(self, master_address):
        self.master_address = master_address

        self.nn_client = None
        self.nn_client_host = ''
        self.nn_class_name = ''

        self.context = None
        self.zmq_client = None

    def run(self):
        self.context = zmq.Context()

        try:
            self._handle_connections()
        except KeyboardInterrupt:
            logging.error('Keyboard Interrupt, shutting down server...')
        finally:
            self.nn_client.stop()
            self.context.term()

    def _handle_connections(self):
        last_evaluations = []

        while True:
            self._connect_client()

            try:
                self.zmq_client.send_pyobj(self.WorkRequest(last_evaluations))
                response = self.zmq_client.recv(self.TIMEOUT)
                if response:
                    last_evaluations = self._handle_response(response)
                else:
                    raise zmq.ZMQError()
            except zmq.ZMQError:
                self._disconnect_client()

    def _handle_response(self, response):
        message = pickle.loads(response)
        if isinstance(message, self.WaitResponse):
            time.sleep(message.wait_time)
            return []
        if isinstance(message, self.WorkResponse):
            self._prepare_network(message.nn_name, message.weights_zip_binary)
            return self._play_games(message.n_games, message.board_states, message.simulations_per_turn)

        return []

    def _prepare_network(self, nn_class_name, weights_zip_binary):
        if self.nn_class_name != nn_class_name:
            self._restart_network(nn_class_name)

        if weights_zip_binary:
            self.nn_client.load_weights(weights_zip_binary)

    def _restart_network(self, nn_class_name):
        self.nn_class_name = nn_class_name
        if self.nn_client:
            self.nn_client.shutdown_server()
            self.nn_client.stop()
            time.sleep(15)

        nn_client.start_nn_server(definitions.SELFPLAY_NN_SERVER_PORT, self.nn_class_name)
        self.nn_client_host = 'tcp://localhost:{}'.format(definitions.SELFPLAY_NN_SERVER_PORT)
        self.nn_client = nn_client.NeuralNetworkClient(self.nn_client_host)
        self.nn_client.start()

    def _play_games(self, n_games, board_states, simulations_per_turn):
        selfplay_pool = multiprocessing.Pool(multiprocessing.cpu_count() - 1)

        results = []
        for _ in range(n_games):
            nn_executor_client = nn_client.NeuralNetworkClient(self.nn_client_host)
            board_state = np.random.choice(board_states)
            game_state = GameState(board_state)

            params = (game_state, nn_executor_client, simulations_per_turn)
            result = selfplay_pool.apply_async(SelfplayServer.play_game, params, callback=SelfplayServer.callback)
            results.append(result)

        evaluations = []
        for result in results:
            evaluations.append(result.get())

        selfplay_pool.terminate()
        selfplay_pool.join()

        return evaluations

    @staticmethod
    def play_game(game_state, nn_executor_client, n_simulations):
        nn_executor_client.start()
        selfplay_executor = core.SelfplayExecutor(game_state, nn_executor_client, n_simulations)
        result = selfplay_executor.run()
        nn_executor_client.stop()
        return result

    @staticmethod
    def callback(result):
        logging.info('Selfplay-Game finished.')

    def _connect_client(self):
        if not self.zmq_client:
            self.zmq_client = self.context.socket(zmq.REQ)
            self.zmq_client.connect(self.master_address)

    def _disconnect_client(self):
        if self.zmq_client:
            self.zmq_client.setsockopt(zmq.LINGER, 0)
            self.zmq_client.close()

# class ParallelSelfplayEvaluationPool:
#     """Parallel execution of selfplay evaluation games using multiple processes on this host machine."""
#     def __init__(self, maps, neural_network_one, neural_network_two, model_file_one, model_file_two,
#                  n_games, simulations_per_turn=100, batch_size=4, pool_size=multiprocessing.cpu_count(),
#                  port_one=6001, port_two=6002):
#         self.nn_server_one = core.NeuralNetworkExecutorServer(neural_network_one, model_file_one,
#                                                               batch_size=batch_size, port=port_one)
#         self.nn_server_two = core.NeuralNetworkExecutorServer(neural_network_two, model_file_two,
#                                                               batch_size=batch_size, port=port_two)
#
#         self.port_one = port_one
#         self.port_two = port_two
#
#         self.n_games = n_games
#         self.n_simulations = simulations_per_turn
#         self.maps = maps
#
#         self.pool_size = pool_size
#
#     def run(self):
#         self.nn_server_one.start()
#         self.nn_server_two.start()
#
#         selfplay_pool = multiprocessing.Pool(processes=self.pool_size)
#         games_per_tournament = max(round(self.n_games / self.pool_size), 1)
#
#         parameters =[]
#         for _ in range(self.pool_size):
#             nn_client_one = core.NeuralNetworkExecutorClient('tcp://localhost:{}'.format(self.port_one))
#             nn_client_two = core.NeuralNetworkExecutorClient('tcp://localhost:{}'.format(self.port_two))
#
#             parameters.append((self.maps, games_per_tournament, self.n_simulations, nn_client_one, nn_client_two))
#
#         results = selfplay_pool.map(ParallelSelfplayEvaluationPool.play_tournament, parameters)
#         selfplay_pool.terminate()
#         selfplay_pool.join()
#
#         result_sum = [0, 0]
#         for result in results:
#             result_sum[0] = result_sum[0] + result[0]
#             result_sum[1] = result_sum[1] + result[1]
#
#         actual_games = self.pool_size * games_per_tournament
#         avg_result = [result_sum[0] / actual_games, result_sum[1] / actual_games]
#
#         self.nn_server_one.stop()
#         self.nn_server_two.stop()
#
#         return avg_result
#
#     @staticmethod
#     def play_tournament(args):
#         maps, n_games, n_simulations, nn_client_one, nn_client_two = args
#
#         nn_client_one.start()
#         nn_client_two.start()
#
#         tournament = core.ModelEvaluator(nn_client_one, nn_client_two, maps)
#         result = tournament.run(n_games, n_simulations)
#
#         nn_client_one.stop()
#         nn_client_two.stop()
#
#         return result
#
#
# class ParallelAITrivialPool:
#     """Parallel execution of tournament games against ai trivial using multiple processes on this host machine."""
#
#     def __init__(self, maps, neural_network, model_file, n_games, time, batch_size=4,
#                  pool_size=multiprocessing.cpu_count(), port=60012):
#         self.maps = maps
#         self.n_games = n_games
#         self.time = time
#         self.pool_size = pool_size
#         self.port = port
#
#         self.nn_server = core.NeuralNetworkExecutorServer(neural_network, model_file, batch_size=batch_size, port=port)
#
#     def run(self):
#         self.nn_server.start()
#
#         selfplay_pool = multiprocessing.Pool(processes=self.pool_size)
#         games_per_tournament = max(round(self.n_games / self.pool_size), 1)
#
#         parameters = []
#         for _ in range(self.pool_size):
#             nn_client = core.NeuralNetworkExecutorClient('tcp://localhost:{}'.format(self.port))
#             parameters.append((self.maps, games_per_tournament, self.time, nn_client))
#
#         results = selfplay_pool.map(ParallelAITrivialPool.play_tournament, parameters)
#         selfplay_pool.terminate()
#         selfplay_pool.join()
#
#         scores_sum = [0, 0]
#         stones_sum = [0, 0]
#         for result in results:
#             for i in range(2):
#                 scores_sum[i] = scores_sum[i] + result[0][i]
#                 stones_sum[i] = stones_sum[i] + result[1][i]
#
#         self.nn_server.stop()
#
#         return scores_sum, stones_sum
#
#     @staticmethod
#     def play_tournament(args):
#         maps, n_games, time, nn_client = args
#
#         nn_client.start()
#         tournament = core.AITrivialEvaluator(nn_client, maps)
#         result = tournament.run(n_games, time)
#         nn_client.stop()
#
#         return result
