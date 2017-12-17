from reversi.game_core import GameState, Field
import reversi.network_core as network
import reversi.tournament as tournament
import reversi.copy as copy
import math
import numpy as np
import threading
import os
import concurrent.futures
import pickle
import random
import time
import definitions
import reinforcement.util as util
import functools


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
        next_game_states = self.game_state.get_next_possible_moves()
        self.possible_moves = [game_state.last_move for game_state in next_game_states]
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

        # Used to embedded smaller boards in big neural networks.
        # The top left position of where the board was inserted
        # has to be stored to reconstruct the outputs after the run.
        self.embedding_position = (0, 0)

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

        for transformation in reversed(self._applied_transformations):
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


class MCTSNode:
    """A single node in the constructed search tree.

    Each node holds its visit count, its value and its direct child nodes."""
    def __init__(self, probability, game_state):
        self.probability = probability
        self.game_state = game_state
        self.visits = 0
        self.total_action_value = {player: 0 for player in range(Field.PLAYER_ONE, Field.PLAYER_EIGHT + 1)}
        self.mean_action_value = {player: 0 for player in range(Field.PLAYER_ONE, Field.PLAYER_EIGHT + 1)}
        self.children = None
        self.is_leave = False

        # Needed to allow multithreaded tree search
        self.expand_lock = threading.Lock()
        self.setter_lock = threading.Lock()

        # Virtual losses...
        self.virtual_loss = {player: -1 for player in range(Field.PLAYER_ONE, Field.PLAYER_EIGHT + 1)}
        self.undo_virtual_loss = {player: 1 for player in range(Field.PLAYER_ONE, Field.PLAYER_EIGHT + 1)}

    def run_simulation_step(self, nn_client):
        # Add virtual loss
        self._update_action_value(self.virtual_loss)

        try:
            if self.is_leave:
                self._increase_visits()
                self._update_action_value(self.game_state.calculate_scores())

                return self.game_state.calculate_scores()

            if self.is_expanded():
                move = self._select_move()
                child = self.children[move]

                result = child.run_simulation_step(nn_client)

                self._increase_visits()
                self._update_action_value(result)

                return result

            # Expanding needs a lock
            self.expand_lock.acquire()
            # We already expanded in another thread, simply re-run.
            if self.is_expanded():
                self.expand_lock.release()
                return self.run_simulation_step(nn_client)

            # We actually need to expand here, lets go
            next_states = self.game_state.get_next_possible_moves()
            if len(next_states) <= 0:
                self.is_leave = True
                self.expand_lock.release()
                self._update_action_value(self.game_state.calculate_scores())
            else:
                evaluation = nn_client.evaluate_game_state(self.game_state)
                self._expand(evaluation, next_states)
                self.expand_lock.release()
                self._update_action_value(evaluation.expected_result)

            self._increase_visits()
            self.mean_action_value = copy.deepcopy(self.total_action_value)
            return self.total_action_value
        finally:
            # Remove virtual loss
            self._update_action_value(self.undo_virtual_loss)

    def _expand(self, evaluation: Evaluation, next_states):
        if self.children:
            return

        self.children = dict()
        for next_state in next_states:
            move = next_state.last_move
            self.children[move] = MCTSNode(evaluation.probabilities[move], next_state)

    def _select_move(self):
        # Select a move using the variant of the PUCT algorithm
        # TODO: Do more research on this algorithm

        # this equals 'sqrt(sum(visit count of children))'
        sqrt_total_child_visits = math.sqrt(self.visits)
        # constant determining exploration
        # TODO: alter this value to find good fit
        c_puct = 3

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
        with self.setter_lock:
            for player, value in new_action_value.items():
                self.total_action_value[player] = self.total_action_value[player] + value
                self.mean_action_value[player] = self.total_action_value[player] / max(self.visits, 1)

    def _increase_visits(self):
        with self.setter_lock:
            self.visits = self.visits + 1

    def is_expanded(self):
        return not not self.children


class MCTSExecutor:
    """Handles the simulation of MCTS in one specific game state.

    This excludes actually progressing the game. The sole purpose of this class
    is to run a specific number of simulation steps starting at a given game state.

    It returns the target move probabilities and the target value of the given game sate."""
    def __init__(self, game_state, nn_client, root_node: MCTSNode=None, thread_pool=None):
        self.nn_client = nn_client
        self.start_game_state = game_state
        self.root_node = root_node

        self.thread_pool = thread_pool

    def run(self, n_simulations):
        if not self.root_node:
            self.root_node = MCTSNode(1.0, self.start_game_state)

        # We can run serial or parallel in a thread pool
        if not self.thread_pool:
            for i in range(n_simulations):
                self.root_node.run_simulation_step(self.nn_client)
        else:
            futures = []
            for i in range(n_simulations):
                futures.append(self.thread_pool.submit(self.root_node.run_simulation_step, self.nn_client))
            concurrent.futures.wait(futures)

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
        thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=8)

        while True:
            # Attach our thread pool to the current executor
            self.current_executor.thread_pool = thread_pool

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

        thread_pool.shutdown(wait=False)
        return self.evaluations

    def _create_evaluation(self):
        """Creates an evaluation of the current MCTSExecutor and adds it to self.evaluations."""
        evaluation = Evaluation(self.current_game_state)
        evaluation.probabilities = self.current_executor.move_probabilities(1.0)

        self.evaluations.append(evaluation)


class TrainingExecutor:
    """Manages the training process of a neural network.

    This is managing the training data and the training process.
    The class is given neural network client to work with.

    The training history size indicates how many of the last games to consider
    for training (e.g. use the 500 most recent games of training data)."""
    def __init__(self, nn_executor_client, data_dir, training_history_size):
        super().__init__()
        self.nn_executor_client = nn_executor_client
        self.training_history_size = training_history_size

        # We will keep the training and test data in a local folder.
        # This class is only responsible for somehow doing the training,
        # this does not constrain it to run only on this machine,
        # but its a good start to have all training data somewhere for whatever training method.
        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        self.lock = threading.Lock()
        self._cache = dict()
        self._current_number = util.count_files(self.data_dir)

    def add_examples(self, evaluations):
        with self.lock:
            self._current_number = self._current_number + 1

            with open(os.path.join(self.data_dir, "{0:010d}.pickle".format(self._current_number)), 'wb') as file:
                pickle.dump(evaluations, file)

    def get_examples(self, n_examples):
        with self.lock:
            evaluations = []
            while len(evaluations) < n_examples:
                oldest_index = max(1, self._current_number - self.training_history_size)
                number = random.randint(oldest_index, self._current_number)
                loaded_evaluations = self._cache.get(number, None)

                if not loaded_evaluations:
                    try:
                        loaded_evaluations = self._load_example(number)
                    except IOError:
                        continue

                # TODO: Better manage the cache
                # FIXME: Delete old elements from cache
                self._cache[number] = loaded_evaluations

                random.shuffle(loaded_evaluations)
                end_index = min(round(n_examples / 8 + 1), len(loaded_evaluations))
                evaluations = evaluations + loaded_evaluations[:end_index]

            return evaluations

    @functools.lru_cache(maxsize=256)
    def _load_example(self, example_number):
        with open(os.path.join(self.data_dir, "{0:010d}.pickle".format(example_number)), 'rb') as file:
            return pickle.load(file)

    def load(self, filename):
        with open(filename, 'rb') as file:
            weights_zip_binary = file.read()
            self.nn_executor_client.load_weights(weights_zip_binary)

    def save(self, filename):
        weights_zip_binary = self.nn_executor_client.save_weights()
        with open(filename, 'wb') as file:
            file.write(weights_zip_binary)

    def run_training_batch(self, batch_size=32):
        # Skip if there is no data
        if self._current_number <= 0:
            time.sleep(10)
            return

        evaluations = self.get_examples(batch_size)
        self.nn_executor_client.execute_training_batch(evaluations)

    def log_loss(self, epoch, batch_size=32):
        raise NotImplementedError()


class ModelEvaluator:
    """Compares two neural network configurations by playing out a small tournament."""
    def __init__(self, nn_client_one, nn_client_two):
        self.nn_client_one = nn_client_one
        self.nn_client_two = nn_client_two

    def run(self, start_game_state, n_simulations):
        thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=8)

        current_game_state = start_game_state

        # TODO: Make universal for more then two players
        tmp = [Field.PLAYER_ONE, Field.PLAYER_TWO]
        np.random.shuffle(tmp)

        player_mapping = {tmp[0]: 0, tmp[1]: 1}

        while True:
            current_player = current_game_state.calculate_next_player()
            if current_player is None:
                break

            # Find the correct nn to execute this move
            if player_mapping[current_player] == 0:
                nn_client = self.nn_client_one
            else:
                nn_client = self.nn_client_two

            # Run the actual simulation to find a move
            mcts_executor = MCTSExecutor(current_game_state, nn_client, thread_pool=thread_pool)
            mcts_executor.run(n_simulations)

            # Find the best move
            selected_move = None
            best_probability = -1.0

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

        thread_pool.shutdown(wait=False)
        return result


class AITrivialEvaluator:
    """Compares a neural network to ai trivial by playing out a small tournament."""
    def __init__(self, nn_client):
        self.nn_client = nn_client

    def run(self, start_game_state, turn_time):
        thread_pool = concurrent.futures.ThreadPoolExecutor(max_workers=8)

        current_game_state = start_game_state

        # TODO: Make universal for more then two players
        tmp = [Field.PLAYER_ONE, Field.PLAYER_TWO]
        np.random.shuffle(tmp)

        player_mapping = {tmp[0]: 0, tmp[1]: 1}

        # TODO: Handle Ports Properly
        found_port = False
        while not found_port:
            try:
                port = random.randint(2000, 4000)
                server = network.BasicServer(start_game_state.board, port)
                server.start()
                found_port = True
            except IOError:
                # TODO: Handle Ports Properly
                found_port = False
                print('Port Conflict, retry...')

        tournament.TrivialAIClient(definitions.AI_TRIVIAL_PATH).start('localhost', port)
        group = server.accept_client()
        server.set_player_for_group(group, tmp[1])

        while True:
            current_player = current_game_state.calculate_next_player()
            if not current_player:
                break

            selected_move = None
            # Find the correct nn to execute this move
            if player_mapping[current_player] == 0:
                end_time = time.time() + turn_time

                mcts_executor = MCTSExecutor(current_game_state, self.nn_client, thread_pool=thread_pool)

                while time.time() < end_time:
                    mcts_executor.run(4)

                # Find the best move
                best_probability = -1.0
                for move, probability in mcts_executor.move_probabilities(1).items():
                    if probability > best_probability:
                        best_probability = probability
                        selected_move = move
            else:
                # We give some extra time to the AI-Trivial, as it usually tries
                # to send a response about 200ms before timeout.
                server.send_player_message(current_player, network.MoveRequestMessage(round(turn_time * 1000 + 200), 0))
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


        thread_pool.shutdown(False)
        return result, stones
