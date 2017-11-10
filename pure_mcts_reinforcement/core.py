from reversi.game_core import GameState, Field
import copy
import math
import numpy as np
import tensorflow as tf
import threading
import queue
import os
import sklearn.model_selection as model_selection
import pickle


class Evaluation:
    """Holds a game state its evaluation.

    The evaluation consists of move probabilities and the expected result of the game state.

    Can be transformed to normal form and back.
    In normal form the currently active player is player one (this should make it easier for the nn)."""
    def __init__(self, game_state: GameState):
        self.game_state = game_state

        # Add dummy values for move probabilities and expected game results
        next_moves = self.game_state.get_next_possible_moves()
        self.probabilities = dict()
        for move in next_moves:
            self.probabilities[move.last_move] = 0.0
        self.expected_result = dict()
        for player in game_state.start_players:
            self.expected_result[player] = 0.0

        # Keep it to be able to transform to/from normal form
        if len(next_moves) > 0:
            (self.active_player, _, _) = next_moves[0].last_move
        else:
            self.active_player = None

    def convert_to_normal(self):
        """Converts the evaluation to a form where the next active player is player one."""
        current_active_player = self.game_state.calculate_next_player()

        # Active Player is already player one
        if current_active_player == Field.PLAYER_ONE:
            return

        rotation_amount = self.game_state.board.n_players - current_active_player.to_player_int() + 1

        self._rotate_players(rotation_amount)

    def convert_from_normal(self):
        """Converts the evaluation from normal form to its original form."""
        current_active_player = self.game_state.calculate_next_player()

        # Active Player is already player one
        if current_active_player == self.active_player:
            return

        rotation_amount = self.active_player.to_player_int() - 1
        self._rotate_players(rotation_amount)

    def _rotate_players(self, rotation_amount):
        """Rotate player numbers. Rotation by one means player one will be player two, and so on."""
        # Rotate the pieces on the board
        for i in range(rotation_amount):
            self.game_state.board.execute_inversion()

        n_players = self.game_state.board.n_players
        # Adjust values describing the current game state
        if self.game_state.last_move:
            (player, pos, choice) = self.game_state.last_move
            if isinstance(choice, Field):
                choice = choice.rotate_by(rotation_amount, n_players)
            self.game_state.last_move = (player.rotate_by(rotation_amount, n_players), pos, choice)

        old_player_bombs = copy.deepcopy(self.game_state.player_bombs)
        for player, bombs in old_player_bombs.items():
            self.game_state.player_bombs[player.rotate_by(rotation_amount, n_players)] = bombs

        old_player_overwrites = copy.deepcopy(self.game_state.player_overwrites)
        for player, overwrites in old_player_overwrites.items():
            self.game_state.player_overwrites[player.rotate_by(rotation_amount, n_players)] = overwrites

        old_players = copy.deepcopy(self.game_state.players)
        self.game_state.players = set()
        for player in old_players:
            self.game_state.players.add(player.rotate_by(rotation_amount, n_players))

        if self.game_state._cached_next_player:
            self.game_state._cached_next_player = \
                self.game_state._cached_next_player.rotate_by(rotation_amount, n_players)

        # Rotate evaluation stats
        old_expected_result = copy.deepcopy(self.expected_result)
        for player, result in old_expected_result.items():
            self.expected_result[player.rotate_by(rotation_amount, n_players)] = result

        old_probabilities = copy.deepcopy(self.probabilities)
        self.probabilities = dict()
        for (player, pos, choice), probability in old_probabilities.items():
            self.probabilities[(player.rotate_by(rotation_amount, n_players), pos, choice)] = probability


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

    def execute_batch(self, sess, game_states):
        raise NotImplementedError("Add implementation that takes game_states and executes them as a batch.")

    def train_batch(self, sess, evaluations):
        raise NotImplementedError("Add implementation that executes one batch training step.")

    def save_weights(self, sess, filename):
        raise NotImplementedError("Add implementation that saves the weights of this network to a checkpoint.")

    def load_weights(self, sess, filename):
        raise NotImplementedError("Add implementation that loads the weights of this network to a checkpoint.")

    def log_loss(self, tf_file_writer, evaluations, epoch):
        raise NotImplementedError("Add implementation to write average losses to the stats file and return them.")


class NeuralNetworkExecutor(threading.Thread):
    """Allows the evaluation of a given game state. Coordinates requests from different threads.

    This is essentially a wrapper for a neural network instance with fixed weights
    that sole purpose is to be executed for different game states.

    The class can for example transparently handle batch execution of the network by blocking
    calls to evaluate game states until a full batch is reached."""

    def __init__(self, neural_network: NeuralNetwork, weights_file=None):
        """Configure the NN. This does not create any tf objects. For that the executor has to be started."""
        super().__init__()
        self.neural_network = neural_network
        self.weights_file = weights_file
        self.graph = tf.Graph()

        # TODO: add more sophisticated synchronization to allow batching. It's ok for a first test.
        self.request_queue = queue.Queue(1)
        self.response_queue = queue.Queue(1)

    def run(self):
        with self.graph.as_default():
            self.neural_network.construct_network()
            with tf.Session() as sess:
                self.neural_network.init_network()

                if self.weights_file:
                    self.neural_network.load_weights(sess, self.weights_file)

                # Do work
                while True:
                    game_state = self.request_queue.get()
                    evaluation = self.neural_network.execute_batch(sess, [game_state])[0]
                    self.response_queue.put(evaluation)

    def execute(self, game_state) -> Evaluation:
        """Executes a neural network to evaluate a given game state.

        Returns the evaluation generated by the network.
        Might block for a while because other evaluations are being performed."""

        self.request_queue.put(game_state)
        return self.response_queue.get()


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

    def run_simulation_step(self, nn_executor: NeuralNetworkExecutor, game_state: GameState):
        if self.is_expanded():
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
        for next_state in next_states:
            move = next_state.last_move
            self.children[move] = MCTSNode(evaluation.probabilities[move])

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
                q = 0.5

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
        self.nn_executor = nn_executor
        self.n_simulations_per_move = n_simulations_per_move
        self.evaluations = []
        self.temperature = 2.0

    def run(self):
        while True:
            # Make sure the game is not finished
            next_states = self.current_executor.start_game_state.get_next_possible_moves()
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
            new_game_state = self.current_executor.start_game_state.execute_move(player, pos, choice)

            # Update our executor. We keep the part of the search tree that was selected.
            selected_child = self.current_executor.root_node.children[move]
            self.current_executor = MCTSExecutor(new_game_state, self.nn_executor, selected_child)

        actual_results = self.current_executor.start_game_state.calculate_scores()
        for evaluation in self.evaluations:
            evaluation.expected_result = actual_results

        return self.evaluations

    def _create_evaluation(self):
        """Creates an evaluation of the current MCTSExecutor and adds it to self.evaluations."""
        evaluation = Evaluation(self.current_executor.start_game_state)
        evaluation.probabilities = self.current_executor.move_probabilities(self.temperature)

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

        # TODO: add more sophisticated synchronization to allow batching. It's ok for a first test.
        self.request_queue = queue.Queue(1)
        self.train_response_queue = queue.Queue(1)
        self.test_loss_response_queue = queue.Queue(1)
        self.train_loss_response_queue = queue.Queue(1)
        self.save_response_queue = queue.Queue(1)

    def run(self):
        with self.graph.as_default():
            self.neural_network.construct_network()
            with tf.Session() as sess:
                self.neural_network.init_network()
                self.neural_network.load_weights(sess, self.weights_file)

                # Do work
                while True:
                    (type, args) = self.request_queue.get()
                    if type == 'train':
                        result = self._run_train_batch_internal(sess, args)
                        self.train_response_queue.put(result)
                    elif type == 'log_test_loss':
                        file_writer, batch_size = args
                        result = self._log_test_loss_internal(file_writer, batch_size)
                        self.test_loss_response_queue.put(result)
                    elif type == 'log_train_loss':
                        file_writer, batch_size = args
                        result = self._log_test_loss_internal(file_writer, batch_size)
                        self.train_loss_response_queue.put(result)
                    elif type == 'save':
                        result = self.neural_network.save_weights(sess, args)
                        self.save_response_queue.put(result)

    def _get_n_training(self):
        with self.lock:
            result = self._n_training
        return result

    def _add_to_n_training(self, amount):
        with self.lock:
            result = self._n_training = self._n_training + amount
        return result

    def _get_n_test(self):
        with self.lock:
            result = self._n_test
        return result

    def _add_to_n_test(self, amount):
        with self.lock:
            result = self._n_test = self._n_test + amount
        return result

    def add_examples(self, evaluations):
        train_evals, test_evals = model_selection.train_test_split(evaluations, test_size=0.2)
        for train_eval in train_evals:
            with open(os.path.join(self.training_dir, "{0:010d}.pickle".format(self._get_n_training())), 'wb') as train_file:
                pickle.dump(train_eval, train_file)
            self._add_to_n_training(1)
        for test_eval in test_evals:
            with open(os.path.join(self.test_dir, "{0:010d}.pickle".format(self._get_n_test())), 'wb') as test_file:
                pickle.dump(test_eval, test_file)
            self._add_to_n_test(1)

    def save(self, filename):
        self.request_queue.put(('save', filename))
        return self.save_response_queue.get()

    def run_training_batch(self, batch_size=32):
        self.request_queue.put(('train', batch_size))
        return self.train_response_queue.get()

    def _run_train_batch_internal(self, sess, batch_size):
        eval_numbers = np.random.randint(0, self._n_training, size=batch_size)

        evals = []
        for eval_number in eval_numbers:
            with open(os.path.join(self.training_dir, "{0:010d}.pickle".format(eval_number)), 'rb') as file:
                evals.append(pickle.load(file))

        self.neural_network.train_batch(sess, evals)
        return None

    def log_test_loss(self, file_writer, batch_size=32):
        self.request_queue.put(('log_test_loss', (file_writer, batch_size)))
        return self.test_loss_response_queue.get()

    def _log_test_loss_internal(self, file_writer, batch_size):
        test_numbers = np.random.randint(0, self._n_test, size=batch_size)

        evals = []
        for test_number in test_numbers:
            with open(os.path.join(self.test_dir, "{0:010d}.pickle".format(test_number)), 'rb') as file:
                evals.append(pickle.load(file))

        return self.neural_network.log_loss(file_writer, evals)

    def log_training_loss(self, file_writer, batch_size=32):
        self.request_queue.put(('log_train_loss', (file_writer, batch_size)))
        return self.train_loss_response_queue.get()

    def _log_training_loss_internal(self, file_writer, batch_size):
        eval_numbers = np.random.randint(0, self._n_training, size=batch_size)

        evals = []
        for eval_number in eval_numbers:
            with open(os.path.join(self.training_dir, "{0:010d}.pickle".format(eval_number)), 'rb') as file:
                evals.append(pickle.load(file))

        return self.neural_network.log_loss(file_writer, evals)


class ModelEvaluator:
    """Compares two neural network configurations by playing out a small tournament."""
    def __init__(self, nn_executor_one, nn_executor_two, map_paths):
        self.nn_executor_one = nn_executor_one
        self.nn_executor_two = nn_executor_two
        self.map_paths = map_paths

    def run(self, n_games, n_simulations):
        total_scores = [0, 0]

        for i in range(n_games):
            map_path = np.random.choice(self.map_paths, 1)
            scores = self._play_game(map_path, n_simulations)

            # TODO: Make universal for more then two players
            for j in range(2):
                total_scores[j] = total_scores[j] + scores[j]

        return total_scores

    def _play_game(self, map_path, n_simulations):
        with open(map_path, 'r') as file:
            board = file.read()
        current_game_state = GameState(board)

        # TODO: Make universal for more then two players
        tmp = [Field.PLAYER_ONE, Field.PLAYER_TWO]
        np.random.shuffle(tmp)

        player_mapping = {tmp[0]: 0, tmp[1]: 1}

        while True:
            next_moves = current_game_state.get_next_possible_moves()
            if len(next_moves) == 0:
                break

            # Find the correct nn to execute this move
            (current_player, _, _) = next_moves[0]
            executor = None
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
            for move, probability in mcts_executor.move_probabilities(1):
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
    # TODO: Pass neural networks and maps to be played

    def run(self, n_games):
        raise NotImplementedError("Play a tournament with #n_games. Return the win-rate of the network.")
