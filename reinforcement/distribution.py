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
import kim
import reversi.copy as copy
import json


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
        self.poll = None

        self.process_pool = multiprocessing.Pool(multiprocessing.cpu_count() - 1)

    def run(self):
        self.context = zmq.Context()
        self.poll = zmq.Poller()

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
                self.zmq_client.send_pyobj(self.WorkRequest(last_work_result), flags=zmq.NOBLOCK)

                socks = dict(self.poll.poll(round(self.TIMEOUT * 1000)))
                if socks.get(self.zmq_client) == zmq.POLLIN:
                    response = self.zmq_client.recv()
                    if response:
                        last_work_result = self._handle_response(response)
                    else:
                        raise zmq.ZMQError()
                else:
                    raise zmq.ZMQError()
            except zmq.ZMQError:
                last_work_result = self.EmptyWorkResult()
                self._disconnect_client()
                logging.info('Server connection closed. Waiting {} seconds, then reconnect...'.format(self.TIMEOUT))
                time.sleep(self.TIMEOUT)

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

            self.poll.register(self.zmq_client, zmq.POLLIN)

    def _disconnect_client(self):
        if self.zmq_client:
            self.zmq_client.setsockopt(zmq.LINGER, 0)
            self.zmq_client.close()
            self.poll.unregister(self.zmq_client)
            self.zmq_client = None


class TrainingMaster:
    """Master Node that coordinates the training process.

    This will order slaves to execute selfplay, selfevaluation and aievaluation games,
    collect the results and use them to train a neural network instance."""
    class State(enum.Enum):
        SELFPLAY = 'SELFPLAY'
        SELFEVAL = 'SELFEVAL'
        AIEVAL = 'AIEVAL'

    DATA_DIR = 'selfplay-data'
    WEIGHTS_DIR = 'weights-history'
    BEST_WEIGHTS = 'best-weights.zip'
    LOG_DIR = 'tensorboard-logs'

    def __init__(self, work_dir, nn_name, start_board_states, port=definitions.TRAINING_MASTER_PORT):
        self.work_dir = work_dir
        if not os.path.exists(self.work_dir):
            os.makedirs(self.work_dir)
        self.weights_dir = os.path.join(self.work_dir, self.WEIGHTS_DIR)
        if not os.path.exists(self.weights_dir):
            os.makedirs(self.weights_dir)
        self.log_dir = os.path.join(self.work_dir, self.LOG_DIR, 'current-run')
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.nn_name = nn_name
        self.port = port

        self.start_board_states = start_board_states

        self.server = None
        self.context = None

        self.training_executor = None
        self.nn_client = None

        # weights cache
        self.current_weights_binary = None
        self.best_weights_binary = None

        # training threads
        self.stopped = False
        self.training_thread_one = None
        self.training_thread_two = None
        self.training_progress_lock = threading.Lock()

        # keep some stats about our progress
        self.progress = TrainingRunProgress(os.path.join(work_dir, 'stats.json'))
        # We could customize settings here, but defaults are fine
        self.progress.load_stats()
        self.progress.save_stats()

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
            self.progress.save_stats()

            self.stopped = True
            self.training_thread_one.join()
            self.training_thread_two.join()

            self.nn_client.stop()
            self.server.close()
            self.context.term()

    def _setup_nn(self):
        nn_client.start_nn_server(definitions.TRAINING_NN_SERVER_PORT, self.nn_name,
                                  log_dir=self.log_dir, start_batch=self.progress.stats.progress.current_batch)
        self.nn_client = nn_client.NeuralNetworkClient('tcp://localhost:{}'.format(definitions.TRAINING_NN_SERVER_PORT))
        self.nn_client.start()

        if self.progress.stats.progress.iteration > 0:
            checkpoint_name = 'checkpoint-{0:05d}.zip'.format(self.progress.stats.progress.iteration)
            with open(os.path.join(self.weights_dir, checkpoint_name), 'rb') as file:
                weights_binary = file.read()
                self.nn_client.load_weights(weights_binary)

            with open(os.path.join(self.work_dir, 'best-checkpoint.zip'), 'rb') as file:
                self.best_weights_binary = file.read()
        else:
            self.progress.stats.progress.iteration = 1
            self.best_weights_binary = self.nn_client.save_weights()
            self._save_best_weights()

        self.current_weights_binary = self.nn_client.save_weights()
        self._save_current_weights()
        self.progress.save_stats()

    def _setup_training_executor(self):
        self.training_executor = core.TrainingExecutor(self.nn_client, os.path.join(self.work_dir, self.DATA_DIR),
                                                       self.progress.stats.settings.training_history_size)

    def _setup_training_threads(self):
        self.training_thread_one = threading.Thread(target=self._run_training_loop)
        self.training_thread_one.start()
        self.training_thread_two = threading.Thread(target=self._run_training_loop)
        self.training_thread_two.start()

    def _run_training_loop(self):
        while not self.stopped:
            self.training_executor.run_training_batch(self.progress.stats.settings.batch_size)
            self._add_training_progress()

    def _add_training_progress(self):
        with self.training_progress_lock:
            self.progress.stats.progress.current_batch += 1
            if self.progress.stats.progress.current_batch % 100 == 0:
                self.progress.save_stats()

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
        simulations_per_turn = self.progress.stats.settings.simulations_per_turn
        epoch = self.progress.stats.progress.iteration
        turn_time = self.progress.stats.settings.turn_time
        board_states = self.start_board_states

        if self.progress.stats.progress.state == self.State.SELFPLAY:
            self.server.send_pyobj(
                PlayingSlave.SelfplayWorkResponse(n_games, self.nn_name, self.best_weights_binary, board_states,
                                                  simulations_per_turn))
        if self.progress.stats.progress.state == self.State.SELFEVAL:
            self.server.send_pyobj(
                PlayingSlave.SelfEvaluationWorkResponse(n_games, self.nn_name, self.current_weights_binary,
                                                        self.best_weights_binary, board_states, simulations_per_turn,
                                                        epoch))
        if self.progress.stats.progress.state == self.State.AIEVAL:
            self.server.send_pyobj(
                PlayingSlave.AIEvaluationWorkResponse(n_games, self.nn_name, self.best_weights_binary, board_states,
                                                      turn_time, epoch))

    def _handle_selfplay_result(self, work_result):
        n_evaluations = len(work_result.evaluation_lists)

        self.progress.stats.current_epoch().self_play.n_games += n_evaluations
        if self.progress.stats.progress.add_samples(self.State.SELFPLAY, n_evaluations):
            # Progress to the next step of the training
            self.progress.stats.progress.n_remaining = self.progress.stats.settings.n_self_eval
            self.progress.stats.progress.state = self.State.SELFEVAL

            # Save a snapshot of the current weights
            self.current_weights_binary = self.nn_client.save_weights()
            self._save_current_weights()

            # Collect some stats on the current training step
            self.progress.stats.current_epoch().self_play.end_batch = self.progress.stats.progress.current_batch
            self.progress.stats.current_epoch().self_eval.start_batch = self.progress.stats.progress.current_batch

            logging.info('Start Self-Evaluation for Iteration {}...'.format(self.progress.stats.progress.iteration))

        for evaluations in work_result.evaluation_lists:
            self.training_executor.add_examples(evaluations)

    def _save_current_weights(self):
        checkpoint_name = 'checkpoint-{0:05d}.zip'.format(self.progress.stats.progress.iteration)
        with open(os.path.join(self.weights_dir, checkpoint_name), 'wb') as file:
            file.write(self.current_weights_binary)

    def _save_best_weights(self):
        with open(os.path.join(self.work_dir, 'best-checkpoint.zip'), 'wb') as file:
            file.write(self.best_weights_binary)

    def _handle_selfeval_result(self, work_result):
        self.progress.stats.current_epoch().self_eval.n_games += work_result.n_games
        self.progress.stats.current_epoch().self_eval.old_score += work_result.nn_two_score
        self.progress.stats.current_epoch().self_eval.new_score += work_result.nn_one_score

        if self.progress.stats.progress.add_samples(self.State.SELFEVAL, work_result.n_games):
            # Progress to the next step of the training
            self.progress.stats.progress.n_remaining = self.progress.stats.settings.n_ai_eval
            self.progress.stats.progress.state = self.State.AIEVAL

            # Collect some stats on the current training step
            self.progress.stats.current_epoch().self_eval.end_batch = self.progress.stats.progress.current_batch
            self.progress.stats.current_epoch().ai_eval.start_batch = self.progress.stats.progress.current_batch

            # See if the new nn was better
            old_score = self.progress.stats.current_epoch().self_eval.old_score
            new_score = self.progress.stats.current_epoch().self_eval.new_score
            n_games = self.progress.stats.current_epoch().self_eval.n_games

            logging.info('Finishing selfplay with result {} vs. {}'
                         .format(new_score, old_score))
            if new_score / n_games > self.progress.stats.settings.needed_avg_score:
                logging.info('Choosing new weights, as it scored better then the current best.')
                self.progress.stats.current_epoch().self_eval.new_better = True
                self.best_weights_binary = self.current_weights_binary

                self._save_best_weights()
            else:
                logging.info('Choosing old weights, as the new ones where not better.')
                self.progress.stats.current_epoch().self_eval.new_better = False

            logging.info('Start AI-Evaluation for Iteration {}...'.format(self.progress.stats.progress.iteration))

    def _handle_aieval_result(self, work_result):
        self.progress.stats.current_epoch().ai_eval.n_games += work_result.n_games
        self.progress.stats.current_epoch().ai_eval.ai_score += work_result.ai_score
        self.progress.stats.current_epoch().ai_eval.nn_score += work_result.nn_score
        self.progress.stats.current_epoch().ai_eval.ai_stones += work_result.ai_stones
        self.progress.stats.current_epoch().ai_eval.nn_stones += work_result.nn_stones

        if self.progress.stats.progress.add_samples(self.State.AIEVAL, work_result.n_games):
            self.progress.stats.current_epoch().ai_eval.end_batch = self.progress.stats.progress.current_batch

            # Progress to the next step of the training
            self.progress.stats.progress.n_remaining = self.progress.stats.settings.n_self_play
            self.progress.stats.progress.state = self.State.SELFPLAY
            self.progress.stats.progress.iteration += 1

            logging.info('Start Selfplay for Iteration {}...'.format(self.progress.stats.progress.iteration))
            self._save_current_weights()
            self.progress.stats.current_epoch().self_play.start_batch = self.progress.stats.progress.current_batch


class TrainingRunProgress:
    """Captures all important information about one training run.

    This includes the settings for the run as well as it's progress.
    This can also be used to coordinate the run itself, including continuing a run."""
    def __init__(self, stats_file_name):
        self.stats_file_name = stats_file_name
        self.stats = TrainingRunStats()
        self.stats.progress.n_remaining = self.stats.settings.n_self_play

    def save_stats(self):
        # Store the enum as an string.
        # Could be cleaned up by using an appropriate mapper.
        to_save = copy.deepcopy(self.stats)
        to_save.progress.state = self.stats.progress.state.value

        mapper = TrainingRunStatsMapper(obj=to_save)
        json_string = json.dumps(mapper.serialize(), indent=4)

        with open(self.stats_file_name + '-copy', 'w') as stats_file:
            stats_file.write(json_string)
        with open(self.stats_file_name, 'w') as stats_file:
            stats_file.write(json_string)

    def load_stats(self):
        if os.path.isfile(self.stats_file_name):
            with open(self.stats_file_name, 'r') as stats_file:
                json_data = json.loads(stats_file.read())
                mapper = TrainingRunStatsMapper(data=json_data)
                loaded = mapper.marshal()
                loaded.progress.state = TrainingMaster.State(loaded.progress.state)

                self.stats = loaded


class TrainingRunStats:
    """Data container for the actual training run stats."""
    class Settings:
        def __init__(self):
            """Init's with default settings. Overwrite if needed."""
            # Number of game states for one training batch
            self.batch_size = 64
            # Number of last games used for training
            self.training_history_size = 128
            # Simulations per selfplay/selfeval turn
            self.simulations_per_turn = 128
            # Turn time for each player during ai evaluation
            self.turn_time = 1.0

            # Number of selfplay games for each iteration
            self.n_self_play = 42
            # Number of self evaluation games for each iteration
            self.n_self_eval = 21
            # Number of evaluation games against the ai-trivial client for each client
            self.n_ai_eval = 14

            # The self evaluation avg. score needed to see this iteration as new best
            self.needed_avg_score = 0.05

    class Progress:
        def __init__(self):
            """Progress statistics of one training run."""
            # Our current state
            self.state = TrainingMaster.State.SELFPLAY
            # The number of samples of the current state needed to progress to the next state
            self.n_remaining = 0
            # The current iteration
            self.iteration = 0
            # The current batch
            self.current_batch = 0

        def add_samples(self, state, n_samples):
            """Call when new samples arrive. Returns True if progress to the next state should be made."""
            if self.state == state:
                self.n_remaining -= n_samples
                if self.n_remaining <= 0:
                    return True

            return False

    class Iteration:
        class SelfEval:
            pass
        class SelfPlay:
            pass
        class AIEval:
            pass

        def __init__(self):
            self.self_eval = self.SelfEval()
            self.self_eval.n_games = 0
            self.self_eval.old_score = 0
            self.self_eval.new_score = 0
            self.self_eval.start_batch = 0
            self.self_eval.end_batch = 0
            self.self_eval.new_better = False

            self.self_play = self.SelfPlay()
            self.self_play.start_batch = 0
            self.self_play.end_batch = 0
            self.self_play.n_games = 0

            self.ai_eval = self.AIEval()
            self.ai_eval.n_games = 0
            self.ai_eval.start_batch = 0
            self.ai_eval.end_batch = 0
            self.ai_eval.nn_score = 0
            self.ai_eval.ai_score = 0
            self.ai_eval.nn_stones = 0
            self.ai_eval.ai_stones = 0

    def __init__(self):
        self.settings = self.Settings()
        self.progress = self.Progress()
        self.iterations = []

    def current_epoch(self):
        while len(self.iterations) < self.progress.iteration:
            self.iterations.append(self.Iteration())

        return self.iterations[self.progress.iteration - 1]


# Python has NO good json serializer, so we need some boilerplate
class SettingsMapper(kim.Mapper):
    __type__ = TrainingRunStats.Settings
    batch_size = kim.field.Integer()
    training_history_size = kim.field.Integer()
    simulations_per_turn = kim.field.Integer()
    turn_time = kim.field.Float()
    n_self_play = kim.field.Integer()
    n_self_eval = kim.field.Integer()
    n_ai_eval = kim.field.Integer()
    needed_avg_score = kim.field.Float()


class ProgressMapper(kim.Mapper):
    __type__ = TrainingRunStats.Progress
    state = kim.field.String()
    n_remaining = kim.field.Integer()
    iteration = kim.field.Integer()
    current_batch = kim.field.Integer()


class IterationSelfEvalMapper(kim.Mapper):
    __type__ = TrainingRunStats.Iteration.SelfEval
    n_games = kim.field.Integer()
    old_score = kim.field.Integer()
    new_score = kim.field.Integer()
    start_batch = kim.field.Integer()
    end_batch = kim.field.Integer()
    new_better = kim.field.Boolean()


class IterationSelfPlayMapper(kim.Mapper):
    __type__ = TrainingRunStats.Iteration.SelfPlay
    n_games = kim.field.Integer()
    start_batch = kim.field.Integer()
    end_batch = kim.field.Integer()


class IterationAIEvalMapper(kim.Mapper):
    __type__ = TrainingRunStats.Iteration.AIEval
    n_games = kim.field.Integer()
    start_batch = kim.field.Integer()
    end_batch = kim.field.Integer()
    nn_score = kim.field.Integer()
    ai_score = kim.field.Integer()
    nn_stones = kim.field.Integer()
    ai_stones = kim.field.Integer()


class IterationMapper(kim.Mapper):
    __type__ = TrainingRunStats.Iteration
    self_eval = kim.field.Nested(IterationSelfEvalMapper, allow_create=True)
    ai_eval = kim.field.Nested(IterationAIEvalMapper, allow_create=True)
    self_play = kim.field.Nested(IterationSelfPlayMapper, allow_create=True)


class TrainingRunStatsMapper(kim.Mapper):
    __type__ = TrainingRunStats
    settings = kim.field.Nested(SettingsMapper, allow_create=True)
    progress = kim.field.Nested(ProgressMapper, allow_create=True)
    iterations = kim.field.Collection(kim.field.Nested(IterationMapper, allow_create=True), allow_create=True)
