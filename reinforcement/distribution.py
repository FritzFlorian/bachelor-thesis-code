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


class PlayingSlave:
    """Runs selfplay games using a given neural network configuration.

    The selfplay server reports it's results back to a master node.
    The master node coordinates workloads, stores results and configures the worker."""
    TIMEOUT = 30

    class WorkResult:
        pass

    class EmptyWorkResult(WorkResult):
        pass

    class SelfplayWorkResult(WorkResult):
        def __init__(self, evaluations):
            self.evaluations = evaluations

    class SelfEvaluationWorkResult(WorkResult):
        def __init__(self, nn_one_score, nn_two_score, n_games):
            self.nn_one_score = nn_one_score
            self.nn_two_score = nn_two_score
            self.n_games = n_games

    class AIEvaluationWorkResult(WorkResult):
        def __init__(self, nn_score, ai_score, nn_stones, ai_stones, n_games):
            self.nn_score = nn_score
            self.ai_score = ai_score
            self.nn_stones = nn_stones
            self.ai_stones = ai_stones
            self.n_games = n_games

    class WorkRequest:
        def __init__(self, work_result):
            self.work_result = work_result

    class SelfplayWorkResponse:
        def __init__(self, n_games, nn_name, weights_zip_binary, board_states, simulations_per_turn):
            self.n_games = n_games
            self.nn_name = nn_name
            self.weights_zip_binary = weights_zip_binary
            self.board_states = board_states
            self.simulations_per_turn = simulations_per_turn

    class SelfEvaluationWorkResponse:
        def __init__(self, n_games, nn_name, weights_binary_one, weights_binary_two,
                     board_states, simulations_per_turn, epoch):
            self.n_games = n_games
            self.nn_name = nn_name
            self.weights_binary_one = weights_binary_one
            self.weights_binary_two = weights_binary_two
            self.board_states = board_states
            self.simulations_per_turn = simulations_per_turn
            self.epoch = epoch

    class AIEvaluationWorkResponse:
        def __init__(self, n_games, nn_name, weights_zip_binary, board_states, turn_time, epoch):
            self.n_games = n_games
            self.nn_name = nn_name
            self.weights_zip_binary = weights_zip_binary
            self.board_states = board_states
            self.turn_time = turn_time
            self.epoch = epoch

    class WaitResponse:
        def __init__(self, wait_time):
            self.wait_time = wait_time

    def __init__(self, master_address):
        self.master_address = master_address

        self.nn_client_one = None
        self.nn_client_host_one = ''
        self.nn_class_name_one = ''

        self.nn_client_two = None
        self.nn_client_host_two = ''
        self.nn_class_name_two = ''

        self.context = None
        self.zmq_client = None

        self.process_pool = multiprocessing.Pool(multiprocessing.cpu_count() - 1)

    def run(self):
        self.context = zmq.Context()

        try:
            self._handle_connections()
        except KeyboardInterrupt:
            logging.error('Keyboard Interrupt, shutting down server...')
        finally:
            self.process_pool.terminate()
            self.nn_client_one.stop()
            self.nn_client_two.stop()
            self.context.term()

    def _handle_connections(self):
        last_work_result = self.EmptyWorkResult()

        while True:
            self._connect_client()

            try:
                self.zmq_client.send_pyobj(self.WorkRequest(last_work_result))
                response = self.zmq_client.recv(self.TIMEOUT)
                if response:
                    last_work_result = self._handle_response(response)
                else:
                    raise zmq.ZMQError()
            except zmq.ZMQError:
                last_work_result = self.EmptyWorkResult()
                self._disconnect_client()

    def _handle_response(self, response):
        message = pickle.loads(response)
        if isinstance(message, self.WaitResponse):
            time.sleep(message.wait_time)
            return self.EmptyWorkResult()
        if isinstance(message, self.SelfplayWorkResponse):
            self._prepare_networks(message.nn_name, message.weights_zip_binary)
            evaluations = self._play_games(message.n_games, message.board_states, message.simulations_per_turn)
            return self.SelfplayWorkResult(evaluations)
        if isinstance(message, self.SelfEvaluationWorkResponse):
            self._prepare_networks(message.nn_name, message.weights_binary_one, message.weights_binary_two)
            scores = self._self_evaluate(message.n_games, message.board_states, message.simulations_per_turn)
            return self.SelfEvaluationWorkResult(scores[0], scores[1], message.n_games)
        if isinstance(message, self.AIEvaluationWorkResponse):
            self._prepare_networks(message.nn_name, message.weights_zip_binary)
            scores, stones = self._ai_evaluate(message.n_games, message.board_states, message.turn_time)
            return self.AIEvaluationWorkResult(scores[0], scores[1], stones[0], stones[1], message.n_games)

        return self.EmptyWorkResult()

    def _prepare_networks(self, nn_class_name, weights_zip_binary, weights_zip_binary_two=None):
        if self.nn_class_name_one != nn_class_name:
            self._restart_network_one(nn_class_name)
        if self.nn_class_name_two != nn_class_name:
            self._restart_network_two(nn_class_name)

        if weights_zip_binary:
            self.nn_client_one.load_weights(weights_zip_binary)
        if weights_zip_binary_two:
            self.nn_client_two.load_weights(weights_zip_binary_two)

    def _restart_network_one(self, nn_class_name):
        self.nn_class_name_one = nn_class_name
        if self.nn_client_one:
            self.nn_client_one.shutdown_server()
            self.nn_client_one.stop()
            time.sleep(15)

        nn_client.start_nn_server(definitions.SELFPLAY_NN_SERVER_PORT, self.nn_class_name_one)
        self.nn_client_host_one = 'tcp://localhost:{}'.format(definitions.SELFPLAY_NN_SERVER_PORT)
        self.nn_client_one = nn_client.NeuralNetworkClient(self.nn_client_host_one)
        self.nn_client_one.start()

    def _restart_network_two(self, nn_class_name):
        self.nn_class_name_two = nn_class_name
        if self.nn_client_two:
            self.nn_client_two.shutdown_server()
            self.nn_client_two.stop()
            time.sleep(15)

        nn_client.start_nn_server(definitions.SELFEVAL_NN_SERVER_PORT, self.nn_class_name_two)
        self.nn_client_host_two = 'tcp://localhost:{}'.format(definitions.SELFEVAL_NN_SERVER_PORT)
        self.nn_client_two = nn_client.NeuralNetworkClient(self.nn_client_host_two)
        self.nn_client_two.start()

    def _play_games(self, n_games, board_states, simulations_per_turn):
        results = []
        for _ in range(n_games):
            nn_executor_client = nn_client.NeuralNetworkClient(self.nn_client_host_one)
            board_state = np.random.choice(board_states)
            game_state = GameState(board_state)

            params = (game_state, nn_executor_client, simulations_per_turn)
            result = self.process_pool.apply_async(PlayingSlave._play_game, params,
                                                   callback=PlayingSlave.selfplay_callback)
            results.append(result)

        evaluations = []
        for result in results:
            evaluations.append(result.get())

        return evaluations

    @staticmethod
    def _play_game(game_state, nn_executor_client, n_simulations):
        nn_executor_client.start()
        selfplay_executor = core.SelfplayExecutor(game_state, nn_executor_client, n_simulations)
        result = selfplay_executor.run()
        nn_executor_client.stop()
        return result

    @staticmethod
    def selfplay_callback(result):
        logging.info('Selfplay-Game finished.')

    def _self_evaluate(self, n_games, board_states, simulations_per_turn):
        results = []
        for _ in range(n_games):
            nn_executor_client_one = nn_client.NeuralNetworkClient(self.nn_client_host_one)
            nn_executor_client_two = nn_client.NeuralNetworkClient(self.nn_client_host_two)

            board_state = np.random.choice(board_states)
            game_state = GameState(board_state)

            params = (game_state, nn_executor_client_one, nn_executor_client_two, simulations_per_turn)
            result = self.process_pool.apply_async(PlayingSlave._play_evaluation_game, params,
                                                   callback=PlayingSlave.eval_callback)
            results.append(result)

        total_score = [0, 0]
        for result in results:
            score = result.get()
            total_score[0] += score[0]
            total_score[1] += score[1]

        return total_score

    @staticmethod
    def _play_evaluation_game(game_state, nn_executor_one, nn_executor_two, n_simulations):
        nn_executor_one.start()
        nn_executor_two.start()

        model_evaluator = core.ModelEvaluator(nn_executor_one, nn_executor_two)
        result = model_evaluator.run(game_state, n_simulations)

        nn_executor_one.stop()
        nn_executor_two.stop()

        return result

    @staticmethod
    def eval_callback(result):
        logging.info('Evaluation-Game finished. ({}:{})'.format(result[0], result[1]))

    def _ai_evaluate(self, n_games, board_states, turn_time):
        results = []
        for _ in range(n_games):
            nn_executor_client_one = nn_client.NeuralNetworkClient(self.nn_client_host_one)

            board_state = np.random.choice(board_states)
            game_state = GameState(board_state)

            params = (game_state, nn_executor_client_one, turn_time)
            result = self.process_pool.apply_async(PlayingSlave._play_ai_evaluation_game, params,
                                                   callback=PlayingSlave.ai_eval_callback)
            results.append(result)

        total_score = [0, 0]
        total_stones = [0, 0]
        for result in results:
            score, stones = result.get()
            for i in range(2):
                total_score[i] += score[i]
                total_stones[i] += stones[i]

        return total_score, total_stones

    @staticmethod
    def _play_ai_evaluation_game(game_state, nn_executor_one, turn_time):
        nn_executor_one.start()
        ai_evaluator = core.AITrivialEvaluator(nn_executor_one)
        result = ai_evaluator.run(game_state, turn_time)
        nn_executor_one.stop()

        return result

    @staticmethod
    def ai_eval_callback(result):
        score, stones = result
        logging.info('AI-Evaluation-Game finished. ({}:{}, {}:{})'.format(score[0], score[1], stones[0], stones[1]))

    def _connect_client(self):
        if not self.zmq_client:
            self.zmq_client = self.context.socket(zmq.REQ)
            self.zmq_client.connect(self.master_address)

    def _disconnect_client(self):
        if self.zmq_client:
            self.zmq_client.setsockopt(zmq.LINGER, 0)
            self.zmq_client.close()
