from reversi.game_core import GameState, Field, Board
import reversi.network_core as network
import reversi.tournament as tournament
import reversi.copy as copy
import math
import numpy as np
import tensorflow as tf
import multiprocessing
import threading
import queue
import os
import sklearn.model_selection as model_selection
import pickle
import random
import time
import zmq


class Evaluation:
    """Holds a game state its evaluation.

    The evaluation consists of move probabilities and the expected result of the game state.

    Can be transformed to normal form and back.
    In normal form the currently active player is player one (this should make it easier for the nn).

    Can be transformed according to dihedral groups.
    NOTE: These transformations currently only work with quadratic fields!"""
    def __init__(self, game_state: GameState):
        self.game_state = game_state

        # Add dummy values for move probabilities and expected game results
        self.possible_moves = [game_state.last_move for game_state in self.game_state.get_next_possible_moves()]
        self.probabilities = dict()
        for move in self.possible_moves:
            self.probabilities[move] = 0.0
        self.expected_result = dict()
        for player in game_state.start_players:
            self.expected_result[player] = 0.0

        # Keep it to be able to transform to/from normal form
        self.active_player = game_state.calculate_next_player()

        # Keep the applied transformations (rotation, mirroring) to be able to undo them.
        # TODO: write about dihedral groups in the thesis
        self._applied_transformations = []

    def convert_to_normal(self):
        """Converts the evaluation to a form where the next active player is player one.

        Returns the converted evaluation and does not change the original data!
        If the evaluation is already in normal form it is simply returned."""
        current_active_player = self.game_state.calculate_next_player()

        # Active Player is already player one
        if current_active_player == Field.PLAYER_ONE:
            return self

        rotation_amount = self.game_state.board.n_players - Field.to_player_int(current_active_player) + 1
        return self._rotate_players(rotation_amount)

    def convert_from_normal(self):
        """Converts the evaluation from normal form to its original form.

        Returns the converted evaluation and does not change the original data!
        If the evaluation is already in normal form it is simply returned."""
        current_active_player = self.game_state.calculate_next_player()

        # Active Player is already player one
        if current_active_player == self.active_player:
            return self

        rotation_amount = Field.to_player_int(self.active_player) - 1
        return self._rotate_players(rotation_amount)

    def _rotate_players(self, rotation_amount):
        """Rotate player numbers. Rotation by one means player one will be player two, and so on."""
        # Do not mutate our object. This could easily lead to hard to spot bugs.
        result = copy.deepcopy(self)

        # Rotate the pieces on the board
        # TODO: Rotate in one go, not in multiple inversions
        for i in range(rotation_amount):
            result.game_state.board.execute_inversion()

        n_players = self.game_state.board.n_players
        # Adjust values describing the current game state
        if self.game_state.last_move:
            (player, pos, choice) = self.game_state.last_move
            if isinstance(choice, Field):
                choice = Field.rotate_by(choice, rotation_amount, n_players)
            result.game_state.last_move = (Field.rotate_by(player, rotation_amount, n_players), pos, choice)

        for player, bombs in self.game_state.player_bombs.items():
            result.game_state.player_bombs[Field.rotate_by(player, rotation_amount, n_players)] = bombs

        for player, overwrites in self.game_state.player_overwrites.items():
            result.game_state.player_overwrites[Field.rotate_by(player, rotation_amount, n_players)] = overwrites

        result.game_state.players = set()
        for player in self.game_state.players:
            result.game_state.players.add(Field.rotate_by(player, rotation_amount, n_players))

        if self.game_state._cached_next_player:
            result.game_state._cached_next_player = \
                Field.rotate_by(self.game_state._cached_next_player, rotation_amount, n_players)

        # Rotate evaluation stats
        result.expected_result = dict()
        for player, expected in self.expected_result.items():
            result.expected_result[Field.rotate_by(player, rotation_amount, n_players)] = expected

        result.probabilities = dict()
        for (player, pos, choice), probability in self.probabilities.items():
            result.probabilities[(Field.rotate_by(player, rotation_amount, n_players), pos, choice)] = probability

        return result

    def apply_transformation(self, number):
        """Applies one dihedral group transformation given an number between 0 and 6.

        Returns the converted evaluation and does not change the original data!"""
        if number == 0:
            # Identity
            return self
        elif number == 1:
            return self._execute_transformation('rot180')
        elif number == 2:
            return self._execute_transformation('mirror')
        elif number == 3:
            tmp = self._execute_transformation('rot180')
            return tmp._execute_transformation('mirror')
        elif number == 4:
            return self._execute_transformation('rot90')
        elif number == 5:
            tmp = self._execute_transformation('rot90')
            return tmp._execute_transformation('mirror')
        elif number == 6:
            tmp = self._execute_transformation('mirror')
            return tmp._execute_transformation('rot90')
        else:
            raise AttributeError('Transformation number must be between 0 and 6, was {}'.format(number))

    def undo_transformations(self):
        """Undos all previously applied transformations.

        Returns the converted evaluation and does not change the original data!"""
        result = self

        for transformation in self._applied_transformations:
            result = result._undo_transformation(transformation)

        return result

    def _execute_transformation(self, transformation):
        if transformation == 'rot90':
            tmp = self._rotate_by_90()
        elif transformation == 'rot180':
            tmp = self._rotate_by_180()
        elif transformation == 'mirror':
            tmp = self._mirror_vertical()
        else:
            raise AttributeError('Transformation "{}" is not supported!'.format(transformation))

        tmp._applied_transformations.append(transformation)
        return tmp

    def _undo_transformation(self, transformation):
        if transformation == 'rot90':
            tmp = self._rotate_by_270()
        elif transformation == 'rot180':
            tmp = self._rotate_by_180()
        elif transformation == 'mirror':
            tmp = self._mirror_vertical()
        else:
            raise AttributeError('Transformation "{}" is not supported!'.format(transformation))

        tmp._applied_transformations.remove(transformation)
        return tmp

    def _swap_positions(self, swap_function):
        """Swaps all positions in the game_state object according to the mapping of the swap function.

        Returns the converted evaluation and does not change the original data!"""
        height = self.game_state.board.height
        width = self.game_state.board.width

        # Currently only supported for quadratic boards
        assert (height == width)

        result = copy.deepcopy(self)

        for y in range(height):
            for x in range(width):
                # Board
                new_pos = swap_function(x, y, width, height)
                result.game_state.board[new_pos] = self.game_state.board[(x, y)]

        # Move Probabilities
        result.probabilities = dict()
        for (player, (x, y), choice), probability in self.probabilities.items():
            result.probabilities[(player, swap_function(x, y, width, height), choice)] = probability

        # Cached possible moves
        if self.possible_moves:
            result.possible_moves = []
            for (player, (x, y), choice) in self.possible_moves:
                result.possible_moves.append((player, swap_function(x, y, width, height), choice))

        # Last Move
        if self.game_state.last_move:
            last_player, (last_x, last_y), last_choice = self.game_state.last_move
            result.game_state.last_move = last_player, swap_function(last_x, last_y, width, height), last_choice

        return result

    def _mirror_vertical(self):
        def swap_function(x, y, width, height):
            new_x = x
            new_y = height - y - 1
            return new_x, new_y

        return self._swap_positions(swap_function)

    def _rotate_by_180(self):
        def swap_function(x, y, width, height):
            new_x = width - x - 1
            new_y = height - y - 1
            return new_x, new_y

        return self._swap_positions(swap_function)

    def _rotate_by_270(self):
        def swap_function(x, y, width, height):
            new_x = y
            new_y = height - x - 1
            return new_x, new_y

        return self._swap_positions(swap_function)

    def _rotate_by_90(self):
        def swap_function(x, y, width, height):
            new_x = height - y - 1
            new_y = x
            return new_x, new_y

        return self._swap_positions(swap_function)


class NeuralNetwork:
    """Wrapper that represents a single neural network instance.

    This is intended to abstract away the actual creation, training and execution of the neural network.
    This should hopefully also allow to re-use major parts of the code for different network structures.

    The network is not responsible for managing its scope/tensorflow graph, this should be done
    by the code that uses and executes it."""
    def construct_network(self):
        raise NotImplementedError("Add the construction of your custom graph structure.")

    def init_network(self):
        raise NotImplementedError("Run initialisation code for your network.")

    def execute_batch(self, sess, evaluations):
        raise NotImplementedError("Add implementation that takes evaluations and fills them as a batch.")

    def train_batch(self, sess, evaluations):
        raise NotImplementedError("Add implementation that executes one batch training step.")

    def save_weights(self, sess, filename):
        raise NotImplementedError("Add implementation that saves the weights of this network to a checkpoint.")

    def load_weights(self, sess, filename):
        raise NotImplementedError("Add implementation that loads the weights of this network to a checkpoint.")

    def log_loss(self, sess, tf_file_writer, evaluations, epoch):
        raise NotImplementedError("Add implementation to write average losses to the stats file and return them.")


class NeuralNetworkExecutorServer(multiprocessing.Process):
    """The actual process running the NN execution.

    This is separate from the NN executor as it allows the NN executor to be passed to
    different processes (it must be able to be pickled!).
    All this is needed to work around the global interpreter lock of python."""
    def __init__(self, neural_network: NeuralNetwork, weights_file, batch_size=1, port=6001):
        """Configure the NN. This does not create any tf objects. For that the executor has to be started."""
        super().__init__()
        self.neural_network = neural_network
        self.weights_file = weights_file
        self.batch_size = batch_size
        self.graph = None

        self.stop_queue = multiprocessing.Queue()
        self.port = port
        self.socket = None
        self.stopped = False

    class Message:
        def __init__(self, response_id, evaluation_bytes):
            self.response_id = response_id
            self.evaluation = pickle.loads(evaluation_bytes)

        def to_multipart(self):
            return [self.response_id, b'', pickle.dumps(self.evaluation)]

    def run(self):
        # Init Network Code
        context = zmq.Context()
        self.socket = context.socket(zmq.ROUTER)
        self.socket.bind('tcp://*:{}'.format(self.port))

        self.graph = tf.Graph()
        with self.graph.as_default():
            self.neural_network.construct_network()
            with tf.Session() as sess:
                self.neural_network.init_network()

                if self.weights_file:
                    self.neural_network.load_weights(sess, self.weights_file)

                messages = []
                while not self.stopped:
                    # Don't  busy wait all the time
                    time.sleep(0.001)
                    try:
                        message = self.socket.recv_multipart(flags=zmq.NOBLOCK)
                        messages.append(NeuralNetworkExecutorServer.Message(message[0], message[2]))

                        if len(messages) >= self.batch_size:
                            self._execute_batch(sess, messages)
                            messages = []
                    except zmq.ZMQError:
                        # Also execute not full batches if no new data arrived in time
                        if len(messages) >= 1:
                            self._execute_batch(sess, messages)
                            messages = []

                    # TODO: Find better way to stop the working server
                    try:
                        self.stop_queue.get(False)
                        self.stopped = True
                    except queue.Empty:
                        pass

        self.socket.close()
        context.term()

    def stop(self):
        self.stop_queue.put('')

    def _execute_batch(self, sess, messages):
        evaluations = [message.evaluation for message in messages]
        evaluations = self.neural_network.execute_batch(sess, evaluations)

        for i in range(len(messages)):
            messages[i].evaluation = evaluations[i]
            self.socket.send_multipart(messages[i].to_multipart())


class NeuralNetworkExecutorClient:
    """Allows the evaluation of a given game state. Coordinates requests from different threads.

    This is essentially a wrapper for a neural network instance with fixed weights
    that sole purpose is to be executed for different game states.

    The Executor starts a separate process. It only holds references to queues, so it
    can be pickled (and therefore also passed to worker processes).

    The class can for example transparently handle batch execution of the network by blocking
    calls to evaluate game states until a full batch is reached."""
    def __init__(self, address):
        """Configure the NN. This does not create any tf objects. For that the executor has to be started."""
        super().__init__()
        self.address = address

        self.context = None
        self.socket = None

    def stop(self):
        self.socket.close()
        self.context.term()

    def start(self):
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.connect(self.address)

    def execute(self, game_state) -> Evaluation:
        """Executes a neural network to evaluate a given game state.

        Returns the evaluation generated by the network.
        Might block for a while because other evaluations are being performed."""
        # Create the evaluation to send to the NN.
        # We will apply rotations on random instances.
        evaluation = Evaluation(game_state)
        evaluation = evaluation.convert_to_normal()
        evaluation = evaluation.apply_transformation(random.randint(0, 6))

        # Execute it using the neural network process.
        self.socket.send_pyobj(evaluation)
        evaluation = self.socket.recv_pyobj()

        # Undo our random rotation on the instance.
        evaluation = evaluation.undo_transformations()
        evaluation = evaluation.convert_from_normal()

        return evaluation


class MCTSNode:
    """A single node in the constructed search tree.

    Each node holds its visit count, its value and its direct child nodes."""
    def __init__(self, probability):
        self.probability = probability
        self.visits = 0
        self.total_action_value = None
        self.mean_action_value = None
        self.children = None
        self.is_leave = False

    def run_simulation_step(self, nn_executor: NeuralNetworkExecutorClient, game_state: GameState):
        if self.is_leave:
            self.visits = self.visits + 1
            self._update_action_value(game_state.calculate_scores())

            return game_state.calculate_scores()
        elif self.is_expanded():
            move = self._select_move()
            child = self.children[move]

            (player, pos, choice) = move
            new_game_state = game_state.execute_move(player, pos, choice)
            result = child.run_simulation_step(nn_executor, new_game_state)

            self.visits = self.visits + 1
            self._update_action_value(result)

            return result
        else:
            next_states = game_state.get_next_possible_moves()
            if len(next_states) <= 0:
                self.total_action_value = game_state.calculate_scores()
                self.is_leave = True
            else:
                evaluation = nn_executor.execute(game_state)
                self.total_action_value = evaluation.expected_result
                self._expand(evaluation, next_states)

            self.visits = self.visits + 1
            self.mean_action_value = copy.deepcopy(self.total_action_value)
            return self.total_action_value

    def _expand(self, evaluation: Evaluation, next_states):
        self.children = dict()
        # print('====================')
        for next_state in next_states:
            move = next_state.last_move
            # print(evaluation.probabilities[move])
            self.children[move] = MCTSNode(evaluation.probabilities[move])
        # print('====================')

    def _select_move(self):
        # Select a move using the variant of the PUCT algorithm
        # TODO: Do more research on this algorithm

        # this equals 'sqrt(sum(visit count of children))'
        sqrt_total_child_visits = math.sqrt(self.visits)
        # constant determining exploration
        # TODO: alter this value to find good fit
        c_puct = 1.0

        best_move_value = -100.0
        best_move = None
        for move, child in self.children.items():
            (player, _, _) = move

            u = c_puct * child.probability * (sqrt_total_child_visits/(1 + child.visits))
            if child.mean_action_value:
                q = child.mean_action_value[player]
            else:
                # TODO: See how to change the rating of reversi to set this properly
                # This should be 'neutral' for unevaluated nodes
                q = 0.0

            move_value = u + q
            if move_value > best_move_value:
                best_move_value = move_value
                best_move = move

        return best_move

    def _update_action_value(self, new_action_value):
        for player, value in new_action_value.items():
            self.total_action_value[player] = self.total_action_value[player] + value
            self.mean_action_value[player] = self.total_action_value[player] / self.visits

    def is_expanded(self):
        return not not self.children


class MCTSExecutor:
    """Handles the simulation of MCTS in one specific game state.

    This excludes actually progressing the game. The sole purpose of this class
    is to run a specific number of simulation steps starting at a given game state.

    It returns the target move probabilities and the target value of the given game sate."""
    def __init__(self, game_state, nn_executor, root_node: MCTSNode=None):
        self.nn_executor = nn_executor
        self.start_game_state = game_state
        self.root_node = root_node

    def run(self, n_simulations):
        if not self.root_node:
            self.root_node = MCTSNode(1.0)

        for i in range(n_simulations):
            self.root_node.run_simulation_step(self.nn_executor, self.start_game_state)

    def move_probabilities(self, temperature):
        """Returns each move and its probability. Temperature controls how much extreme values are damped."""
        exponent = 1.0 / temperature

        visit_sum = 0
        exponentiated_visit_counts = dict()
        for move, child in self.root_node.children.items():
            exponentiated_visit_count = child.visits ** exponent
            visit_sum = visit_sum + exponentiated_visit_count

            exponentiated_visit_counts[move] = exponentiated_visit_count

        return {move: count / visit_sum for move, count in exponentiated_visit_counts.items()}


class SelfplayExecutor:
    """Handles the simulation one selfplay game.

    This should run one game of selfplay and return a list of all states and all
    corresponding probability/value targets that can then be used as training data."""
    def __init__(self, game_state, nn_executor, n_simulations_per_move):
        self.current_executor = MCTSExecutor(game_state, nn_executor)
        self.current_game_state = copy.deepcopy(game_state)
        self.nn_executor = nn_executor
        self.n_simulations_per_move = n_simulations_per_move
        self.evaluations = []
        self.temperature = 1.0

    def run(self):
        while True:
            # Make sure the game is not finished
            next_states = self.current_game_state.get_next_possible_moves()
            if len(next_states) == 0:
                break

            # Run the simulation
            self.current_executor.run(self.n_simulations_per_move)

            # Take a snapshot for training
            self._create_evaluation()

            # Select next move
            move_probabilities = self.current_executor.move_probabilities(self.temperature).items()
            moves = [item[0] for item in move_probabilities]
            probabilities = [item[1] for item in move_probabilities]

            # Execute the move
            index = np.random.choice(len(moves), p=probabilities)
            (player, pos, choice) = move = moves[index]
            self.current_game_state = self.current_game_state.execute_move(player, pos, choice)

            # Update our executor. We keep the part of the search tree that was selected.
            selected_child = self.current_executor.root_node.children[move]
            self.current_executor = MCTSExecutor(self.current_game_state, self.nn_executor, selected_child)

        actual_results = self.current_game_state.calculate_scores()

        old_evaluations = self.evaluations
        self.evaluations = []
        for evaluation in old_evaluations:
            # Add results the next possible moves
            evaluation.expected_result = actual_results

            # Convert to normal...
            evaluation = evaluation.convert_to_normal()

            # Add every possible rotation
            for i in range(7):
                transformed_evaluation = evaluation.apply_transformation(i)
                self.evaluations.append(transformed_evaluation)

        return self.evaluations

    def _create_evaluation(self):
        """Creates an evaluation of the current MCTSExecutor and adds it to self.evaluations."""
        evaluation = Evaluation(self.current_game_state)
        evaluation.probabilities = self.current_executor.move_probabilities(0.5)

        self.evaluations.append(evaluation)


class TrainingExecutor(threading.Thread):
    """Manages the training process of a neural network.

    This is managing the training set, test set and training process.
    The class is given an initial weight configuration. It then is fed example data.
    It has to manage the example data internally, split it into a training and test set."""
    def __init__(self, neural_network: NeuralNetwork, weights_file, data_dir):
        super().__init__()
        self.neural_network = neural_network
        self.weights_file = weights_file
        self.graph = tf.Graph()

        # We will keep the training and test data in a local folder.
        # This class is only responsible for somehow doing the training,
        # this does not constrain it to run only on this machine,
        # but its a good start to have all training data somewhere for whatever training method.
        self.training_dir = os.path.join(data_dir, 'training')
        self.test_dir = os.path.join(data_dir, 'test')
        if not os.path.exists(self.training_dir):
            os.makedirs(self.training_dir)
        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)

        # TODO: Replace this with a more sophisticated synchronization mechanism
        self.lock = threading.Lock()
        self._n_training = 0
        self._n_test = 0
        self._cached_training = dict()
        self._cached_test = dict()

        # TODO: add more sophisticated synchronization to allow batching. It's ok for a first test.
        self.request_queue = queue.Queue(1)

        self.stopped = False

    def run(self):
        with self.graph.as_default():
            self.neural_network.construct_network()
            with tf.Session() as sess:
                self.neural_network.init_network()
                self.neural_network.load_weights(sess, self.weights_file)

                while not self.stopped:
                    try:
                        event = self.request_queue.get(timeout=1)
                        type = event.type
                        args = event.args

                        if type == 'train':
                            event.result = self._run_train_batch_internal(sess, args)
                        elif type == 'log_test_loss':
                            file_writer, batch_size, epoch = args
                            event.result = self._log_test_loss_internal(sess, file_writer, batch_size, epoch)
                        elif type == 'log_train_loss':
                            file_writer, batch_size, epoch = args
                            event.result = self._log_training_loss_internal(sess, file_writer, batch_size, epoch)
                        elif type == 'save':
                            event.result = self.neural_network.save_weights(sess, args)

                        event.set()
                    except queue.Empty:
                        pass

    def stop(self):
        self.stopped = True

    def add_examples(self, evaluations, test_game=False):
        with self.lock:
            if test_game:
                directory = self.test_dir
                self._n_test = self._n_test + 1
                number = self._n_test
                cache = self._cached_test
            else:
                directory = self.training_dir
                self._n_training = self._n_training + 1
                number = self._n_training
                cache = self._cached_training

            with open(os.path.join(directory, "{0:010d}.pickle".format(number)), 'wb') as file:
                pickle.dump(evaluations, file)

            # TODO: Better manage the cache
            cache[number] = evaluations

    def get_examples(self, n_examples, test_game=False):
        with self.lock:
            if test_game:
                directory = self.test_dir
                max_number = self._n_test
                cache = self._cached_test
            else:
                directory = self.training_dir
                max_number = self._n_training
                cache = self._cached_training

            evaluations = []
            while len(evaluations) < n_examples:
                number = random.randint(1, max_number)
                loaded_evaluations = cache.get(number, None)

                if not loaded_evaluations:
                    try:
                        with open(os.path.join(directory, "{0:010d}.pickle".format(number)), 'rb') as file:
                            loaded_evaluations = pickle.load(file)
                    except IOError:
                        break

                cache[number] = loaded_evaluations

                random.shuffle(loaded_evaluations)
                end_index = min(round(n_examples / 4 + 1), len(loaded_evaluations))
                evaluations = evaluations + loaded_evaluations[:end_index]

            return evaluations

    def save(self, filename):
        event = threading.Event()
        event.type = 'save'
        event.args = filename
        self.request_queue.put(event)
        event.wait()
        return event.result

    def run_training_batch(self, batch_size=32):
        event = threading.Event()
        event.type = 'train'
        event.args = batch_size
        self.request_queue.put(event)
        event.wait()
        return event.result

    def _run_train_batch_internal(self, sess, batch_size):
        evals = self.get_examples(batch_size)
        self.neural_network.train_batch(sess, evals)
        return None

    def log_test_loss(self, file_writer, epoch, batch_size=32):
        event = threading.Event()
        event.type = 'log_test_loss'
        event.args = (file_writer, batch_size, epoch)
        self.request_queue.put(event)
        event.wait()
        return event.result

    def _log_test_loss_internal(self, sess, file_writer, batch_size, epoch):
        evals = self.get_examples(batch_size, test_game=True)
        return self.neural_network.log_loss(sess, file_writer, evals, epoch)

    def log_training_loss(self, file_writer, epoch, batch_size=32):
        event = threading.Event()
        event.type = 'log_train_loss'
        event.args = (file_writer, batch_size, epoch)
        self.request_queue.put(event)
        event.wait()
        return event.result

    def _log_training_loss_internal(self, sess, file_writer, batch_size, epoch):
        evals = self.get_examples(batch_size)
        return self.neural_network.log_loss(sess, file_writer, evals, epoch)


class ModelEvaluator:
    """Compares two neural network configurations by playing out a small tournament."""
    def __init__(self, nn_executor_one, nn_executor_two, map_paths):
        self.nn_executor_one = nn_executor_one
        self.nn_executor_two = nn_executor_two
        self.map_paths = map_paths

    def run(self, n_games, n_simulations):
        total_scores = [0, 0]

        for i in range(n_games):
            map_path = np.random.choice(self.map_paths)
            scores = self._play_game(map_path, n_simulations)

            # TODO: Make universal for more then two players
            for j in range(2):
                total_scores[j] = total_scores[j] + scores[j]

        return total_scores

    def _play_game(self, map_path, n_simulations):
        with open(map_path, 'r') as file:
            board = Board(file.read())
        current_game_state = GameState(board)

        # TODO: Make universal for more then two players
        tmp = [Field.PLAYER_ONE, Field.PLAYER_TWO]
        np.random.shuffle(tmp)

        player_mapping = {tmp[0]: 0, tmp[1]: 1}

        while True:
            game_state_copy = copy.deepcopy(current_game_state)
            current_player = current_game_state.calculate_next_player()
            if current_player is None:
                break

            # Find the correct nn to execute this move
            if player_mapping[current_player] == 0:
                executor = self.nn_executor_one
            else:
                executor = self.nn_executor_two

            # Run the actual simulation to find a move
            mcts_executor = MCTSExecutor(current_game_state, executor)
            mcts_executor.run(n_simulations)

            # Find the best move
            selected_move = None
            best_probability = -1.0
            if not mcts_executor.root_node.children:
                print('===========')
                print(game_state_copy.board.__string__())
                print(current_game_state.board.__string__())
                print(current_game_state._cached_next_player)
                print(current_game_state.calculate_next_player())
                print(current_player)

            for move, probability in mcts_executor.move_probabilities(1).items():
                if probability > best_probability:
                    best_probability = probability
                    selected_move = move

            # Execute the move
            player, pos, choice = selected_move
            current_game_state = current_game_state.execute_move(player, pos, choice)

        # return the scores
        scores = current_game_state.calculate_scores()

        # TODO: Make universal for more then two players
        result = [0, 0]
        result[player_mapping[Field.PLAYER_ONE]] = scores[Field.PLAYER_ONE]
        result[player_mapping[Field.PLAYER_TWO]] = scores[Field.PLAYER_TWO]

        return result


class AITrivialEvaluator:
    """Compares a neural network to ai trivial by playing out a small tournament."""
    def __init__(self, nn_executor, map_paths):
        self.nn_executor = nn_executor
        self.map_paths = map_paths

    def run(self, n_games, time):
        total_scores = [0, 0]
        total_stones = [0, 0]

        for i in range(n_games):
            map_path = np.random.choice(self.map_paths)
            try:
                scores, stones = self._play_game(map_path, time)

                # TODO: Make universal for more then two players
                for j in range(2):
                    total_scores[j] = total_scores[j] + scores[j]
                    total_stones[j] = total_stones[j] + stones[j]
            except IOError:
                # TODO: Handle Ports Properly
                print('Port Conflict...')

        return total_scores, total_stones

    def _play_game(self, map_path, turn_time):
        with open(map_path, 'r') as file:
            board = Board(file.read())
        current_game_state = GameState(board)

        # TODO: Make universal for more then two players
        tmp = [Field.PLAYER_ONE, Field.PLAYER_TWO]
        np.random.shuffle(tmp)

        player_mapping = {tmp[0]: 0, tmp[1]: 1}

        # TODO: Handle Ports Properly
        port = random.randint(2000, 4000)
        server = network.BasicServer(board, port)
        server.start()
        tournament.TrivialAIClient('./ai_trivial').start('localhost', port)
        group = server.accept_client()
        server.set_player_for_group(group, tmp[1])

        while True:
            current_player = current_game_state.calculate_next_player()
            if not current_player:
                break

            selected_move = None
            # Find the correct nn to execute this move
            if player_mapping[current_player] == 0:
                end_time = time.clock() + turn_time

                mcts_executor = MCTSExecutor(current_game_state, self.nn_executor)

                while time.clock() < end_time:
                    mcts_executor.run(2)

                # Find the best move
                best_probability = -1.0
                for move, probability in mcts_executor.move_probabilities(1).items():
                    if probability > best_probability:
                        best_probability = probability
                        selected_move = move
            else:
                server.send_player_message(current_player, network.MoveRequestMessage(round(turn_time * 1000), 0))
                move_response = server.read_player_message(current_player, network.MoveResponseMessage)
                selected_move = (current_player, move_response.pos, move_response.choice)

            # Execute the move
            player, pos, choice = selected_move
            current_game_state = current_game_state.execute_move(player, pos, choice)
            server.broadcast_message(network.MoveNotificationMessage(pos, choice, player))

        # return the scores
        server.broadcast_message(network.EndPhaseOneMessage())
        server.broadcast_message(network.EndPhaseTwoMessage())
        server.stop()
        scores = current_game_state.calculate_scores()

        # TODO: Make universal for more then two players
        result = [0, 0]
        result[player_mapping[Field.PLAYER_ONE]] = scores[Field.PLAYER_ONE]
        result[player_mapping[Field.PLAYER_TWO]] = scores[Field.PLAYER_TWO]

        stones = [0, 0]
        stones[player_mapping[Field.PLAYER_ONE]] = current_game_state.board.count({Field.PLAYER_ONE})[Field.PLAYER_ONE]
        stones[player_mapping[Field.PLAYER_TWO]] = current_game_state.board.count({Field.PLAYER_TWO})[Field.PLAYER_TWO]

        return result, stones
