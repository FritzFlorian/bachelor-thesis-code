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
import enum
import os
import threading
import reinforcement.util as util


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
        def __init__(self, evaluation_lists):
            self.evaluation_lists = evaluation_lists

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
                logging.info('Sending work request to master server...')
                self.zmq_client.send_pyobj(self.WorkRequest(last_work_result))
                response = self.zmq_client.recv(self.TIMEOUT)
                if response:
                    last_work_result = self._handle_response(response)
                else:
                    raise zmq.ZMQError()
            except zmq.ZMQError:
                last_work_result = self.EmptyWorkResult()
                self._disconnect_client()
                logging.info('Server connection closed. Waiting 10 seconds, then reconnect...')
                time.sleep(10)

    def _handle_response(self, response):
        message = pickle.loads(response)
        if isinstance(message, self.WaitResponse):
            time.sleep(message.wait_time)
            return self.EmptyWorkResult()
        if isinstance(message, self.SelfplayWorkResponse):
            self._prepare_networks(message.nn_name, message.weights_zip_binary)
            evaluation_lists = self._play_games(message.n_games, message.board_states, message.simulations_per_turn)
            return self.SelfplayWorkResult(evaluation_lists)
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

        evaluation_lists = []
        for result in results:
            evaluation_lists.append(result.get())

        return evaluation_lists

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


class TrainingMaster:
    """Master Node that coordinates the training process.

    This will order slaves to execute selfplay, selfevaluation and aievaluation games,
    collect the results and use them to train a neural network instance."""
    class State(enum.Enum):
        SELFPLAY = 1
        SELFEVAL = 2
        AIEVAL = 3

    DATA_DIR = 'selfplay-data'
    WEIGHTS_DIR = 'weights-history'
    BEST_WEIGHTS = 'best-weights.zip'

    def __init__(self, work_dir, nn_name, start_board_states, port=definitions.TRAINING_MASTER_PORT):
        self.work_dir = work_dir
        if not os.path.exists(self.work_dir):
            os.makedirs(self.work_dir)
        self.weights_dir = os.path.join(self.work_dir, self.WEIGHTS_DIR)
        if not os.path.exists(self.weights_dir):
            os.makedirs(self.weights_dir)
        self.epoch = util.count_files(self.weights_dir)

        self.nn_name = nn_name
        self.port = port

        self.start_board_states = start_board_states

        self.server = None
        self.context = None

        self.training_executor = None
        self.nn_client = None

        # Config
        self.batch_size = 64
        self.training_history_size = 64
        self.simulations_per_turn = 128
        self.turn_time = 1.0

        self.n_training_batches = 1000
        self.n_self_eval = 21
        self.n_ai_eval = 21

        self.needed_avg_score = 0.05

        # States
        self.state = self.State.SELFPLAY
        self.training_remaining = self.n_training_batches
        self.self_eval_remaining = 0
        self.ai_eval_remaining = 0

        # Stats
        self.self_eval_scores = [0, 0]
        self.n_self_eval_games = 0

        self.ai_eval_scores = [0, 0]
        self.ai_eval_stones = [0, 0]

        # weights cache
        self.current_weights_binary = None
        self.best_weights_binary = None

        # training threads
        self.stopped = False
        self.training_thread_one = None
        self.training_thread_two = None
        self.training_progress_lock = threading.Lock()

    def run(self):
        self.context = zmq.Context()
        self.server = self.context.socket(zmq.REP)
        self.server.bind('tcp://*:{}'.format(self.port))

        self._setup_nn()
        self._setup_training_executor()
        self._setup_training_threads()
        try:
            self._handle_messages()
        except KeyboardInterrupt:
            logging.error('Keyboard Interrupt, shutting down server...')
        finally:
            self.stopped = True
            self.training_thread_one.join()
            self.training_thread_two.join()

            self.nn_client.stop()
            self.server.close()
            self.context.term()

    def _setup_nn(self):
        nn_client.start_nn_server(definitions.TRAINING_NN_SERVER_PORT, self.nn_name)
        self.nn_client = nn_client.NeuralNetworkClient('tcp://localhost:{}'.format(definitions.TRAINING_NN_SERVER_PORT))
        self.nn_client.start()

        if self.epoch > 0:
            checkpoint_name = 'checkpoint-{0:05d}.zip'.format(self.epoch)
            with open(os.path.join(self.weights_dir, checkpoint_name), 'rb') as file:
                weights_binary = file.read()
                self.nn_client.load_weights(weights_binary)

            with open(os.path.join(self.work_dir, 'best-checkpoint.zip'), 'rb') as file:
                self.best_weights_binary = file.read()
        else:
            self.epoch = 0
            self.best_weights_binary = self.nn_client.save_weights()
            with open(os.path.join(self.work_dir, 'best-checkpoint.zip'), 'wb') as file:
                file.write(self.best_weights_binary)

        self.current_weights_binary = self.nn_client.save_weights()

    def _setup_training_executor(self):
        self.training_executor = core.TrainingExecutor(self.nn_client, os.path.join(self.work_dir, self.DATA_DIR),
                                                       self.training_history_size)

    def _setup_training_threads(self):
        self.training_thread_one = threading.Thread(target=self._run_training_loop)
        self.training_thread_one.start()
        self.training_thread_two = threading.Thread(target=self._run_training_loop)
        self.training_thread_two.start()

    def _run_training_loop(self):
        while not self.stopped:
            self.training_executor.run_training_batch(self.batch_size)
            self._add_training_progress(1)

    def _add_training_progress(self, n_batches):
        with self.training_progress_lock:
            self.training_remaining -= n_batches
            if self.training_remaining <= 0 and self.state == self.State.SELFPLAY:
                self.current_weights_binary = self.nn_client.save_weights()
                checkpoint_name = 'checkpoint-{0:05d}.zip'.format(self.epoch)
                with open(os.path.join(self.weights_dir, checkpoint_name), 'wb') as file:
                    file.write(self.current_weights_binary)

                self.self_eval_remaining = self.n_self_eval
                self.state = self.State.SELFEVAL
                self.epoch += 1
                logging.info('Progressing to Epoch {}. Starting Self- and AI-Evaluation...'.format(self.epoch))

    def _handle_messages(self):
        while True:
            request = self.server.recv_pyobj()

            if isinstance(request, PlayingSlave.WorkRequest):
                logging.debug('Work request {}.'.format(request))
                self._handle_work_request(request)
            else:
                self.server.send('unsupported message type')

    def _handle_work_request(self, request: PlayingSlave.WorkRequest):
        work_result = request.work_result

        if isinstance(work_result, PlayingSlave.SelfplayWorkResult):
            self._handle_selfplay_result(work_result)
        if isinstance(work_result, PlayingSlave.SelfEvaluationWorkResult):
            self._handle_selfeval_result(work_result)
        if isinstance(work_result, PlayingSlave.AIEvaluationWorkResult):
            self._handle_aieval_result(work_result)

        self._send_work_response()

    def _send_work_response(self):
        n_games = 7
        nn_name = self.nn_name
        simulations_per_turn = self.simulations_per_turn
        epoch = self.epoch
        turn_time = self.turn_time
        board_states = self.start_board_states

        if self.state == self.State.SELFPLAY:
            self.server.send_pyobj(
                PlayingSlave.SelfplayWorkResponse(n_games, nn_name, self.best_weights_binary, board_states,
                                                  simulations_per_turn))
        if self.state == self.State.SELFEVAL:
            self.server.send_pyobj(
                PlayingSlave.SelfEvaluationWorkResponse(n_games, nn_name, self.current_weights_binary,
                                                        self.best_weights_binary, board_states, simulations_per_turn,
                                                        epoch))
        if self.state == self.State.AIEVAL:
            self.server.send_pyobj(
                PlayingSlave.AIEvaluationWorkResponse(n_games, nn_name, self.best_weights_binary, board_states,
                                                      turn_time, epoch))
            self.ai_eval_remaining -= n_games
            if self.ai_eval_remaining <= 0:
                self.training_remaining = self.n_training_batches
                self.state = self.State.SELFPLAY

    def _handle_selfplay_result(self, work_result):
        for evaluations in work_result.evaluation_lists:
            self.training_executor.add_examples(evaluations)

    def _handle_selfeval_result(self, work_result):
        self.self_eval_remaining -= work_result.n_games
        if self.state == self.State.SELFEVAL:
            self.self_eval_scores[0] += work_result.nn_one_score
            self.self_eval_scores[1] += work_result.nn_two_score
            self.n_self_eval_games += work_result.n_games

        if self.self_eval_remaining <= 0:
            logging.info('Finishing selfplay with result {} vs. {}'
                         .format(self.self_eval_scores[0], self.self_eval_scores[1]))
            if self.self_eval_scores[0] / self.n_self_eval_games > self.needed_avg_score:
                logging.info('Choosing new weights, as it scored better then the current best.')
                self.best_weights_binary = self.current_weights_binary
                with open(os.path.join(self.work_dir, 'best-checkpoint.zip'), 'wb') as file:
                    file.write(self.best_weights_binary)
            else:
                logging.info('Choosing old weights, as the new ones where not better.')

            self.self_eval_scores = [0, 0]
            self.self_eval_remaining = self.n_self_eval
            self.ai_eval_remaining = self.n_ai_eval
            self.state = self.State.AIEVAL

    def _handle_aieval_result(self, work_result):
        print('got ai eval results... TODO: process them')
