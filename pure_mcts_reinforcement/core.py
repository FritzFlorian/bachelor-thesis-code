from reversi.game_core import GameState, Field
import copy


class NeuralNetworkExecutor:
    """Allows the evaluation of a given game state. Coordinates requests from different threads.

    This is essentially a wrapper for a neural network instance with fixed weights
    that sole purpose is to be executed for different game states.

    The class can for example transparently handle batch execution of the network by blocking
    calls to evaluate game states until a full batch is reached."""
    def __init__(self, configured_nn):
        # TODO: How is the network configuration passed? Is it simply an instance?
        # No matter what it is, it's once configured and then stays the same for the
        # lifetime of the object!
        self.configured_nn = configured_nn

    def execute(self, game_state):
        """Executes a neural network to evaluate a given game state.

        Returns the evaluation generated by the network.
        Might block for a while because other evaluations are being performed."""
        raise NotImplementedError("Add network execution.")


class NeuralNetwork:
    """Wrapper that represents a single neural network instance.

    This is intended to abstract away the actual creation, training and execution of the neural network.
    This should hopefully also allow to re-use major parts of the code for different network structures."""
    def construct_network(self):
        raise NotImplementedError("Add the construction of your custom graph structure.")

    def init_network(self):
        raise NotImplementedError("Run initialisation code for your network.")

    def execute_batch(self, game_states):
        raise NotImplementedError("Add implementation that takes game_states and executes them as a batch.")

    def train_batch(self, game_states, outputs):
        raise NotImplementedError("Add implementation that executes one batch training step.")

    def save_weights(self, filename):
        raise NotImplementedError("Add implementation that saves the weights of this network to a checkpoint.")

    def load_weights(self, filename):
        raise NotImplementedError("Add implementation that loads the weights of this network to a checkpoint.")

    def log_loss(self, tf_file_writer, game_states, outputs):
        raise NotImplementedError("Add implementation to write average losses to the stats file and return them.")


class MCTSExecutor:
    """Handles the simulation of MCTS in one specific game state.

    This excludes actually progressing the game. The sole purpose of this class
    is to run a specific number of simulation steps starting at a given game state.

    It then returns the target move probabilities and the target value of the given game sate."""
    # TODO: Initialisation and Reuse of older runs
    # TODO: pass game state and NNExecutor on init

    def run(self, n_simulations):
        raise NotImplementedError("Run #n_simulations simulation steps on the tree and return an target evaluation.")


class SelfplayExecutor:
    """Handles the simulation one selfplay game.

    This should run one game of selfplay and return a list of all states and all
    corresponding probability/value targets that can then be used as training data."""
    # TODO: pass game state and NNExecutor on init
    # TODO: pass simulation runs

    def run(self):
        raise NotImplementedError("Run simulation and return target evaluations.")


class TrainingExecutor:
    """Manages the training process of a neural network.

    This is managing the training set, test set and training process.
    The class is given an initial weight configuration. It then is fed example data.
    It has to manage the example data internally, split it into a training and test set."""
    def add_examples(self, evaluations):
        raise NotImplementedError("Add the given examples to be managed and used for training.")

    def run_training_batch(self):
        raise NotImplementedError("Run a single training batch.")


class ModelEvaluator:
    """Compares two neural network configurations by playing out a small tournament."""
    # TODO: Pass neural networks and maps to be played

    def run(self, n_games):
        raise NotImplementedError("Play a tournament with #n_games. Return the win-rate of the new network.")


class AITrivialEvaluator:
    """Compares a neural network to ai trivial by playing out a small tournament."""
    # TODO: Pass neural networks and maps to be played

    def run(self, n_games):
        raise NotImplementedError("Play a tournament with #n_games. Return the win-rate of the network.")


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
            (_, pos, choice) = move.last_move
            self.probabilities[(pos, choice)] = 0.0
        self.expected_result = dict()
        for player in game_state.players:
            self.expected_result[player] = 0.0

        # Keep it to be able to transform to/from normal form
        (self.active_player, _, _) = next_moves[0].last_move

    def convert_to_normal(self):
        """Converts the evaluation to a form where the next active player is player one."""
        next_moves = self.game_state.get_next_possible_moves()
        (current_active_player, _, _) = next_moves[0].last_move

        # Active Player is already player one
        if current_active_player == Field.PLAYER_ONE:
            return

        rotation_amount = self.game_state.board.n_players - current_active_player.to_int() + 1
        self._rotate_players(rotation_amount)

    def convert_from_normal(self):
        """Converts the evaluation from normal form to its original form."""
        next_moves = self.game_state.get_next_possible_moves()
        (current_active_player, _, _) = next_moves[0].last_move

        # Active Player is already player one
        if current_active_player == self.active_player:
            return

        rotation_amount = self.active_player.to_int() - 1
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

        # Rotate evaluation stats
        old_expected_result = copy.deepcopy(self.expected_result)
        for player, result in old_expected_result.items():
            self.expected_result[player.rotate_by(rotation_amount, n_players)] = result
