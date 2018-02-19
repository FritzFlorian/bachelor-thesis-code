import hometrainer.core
import hometrainer.util
from reversi.game_core import GameState, Field


class ReversiMove(hometrainer.core.Move):
    """The reversi Module uses a tuple as a move, simply wrap it to make it work with the hometrainer library."""
    def __init__(self, internal_tuple):
        self.internal_tuple = internal_tuple

    def __eq__(self, other):
        return self.internal_tuple == other.internal_tuple

    def __hash__(self):
        return hash(self.internal_tuple)


class ReversiGameState(hometrainer.core.GameState):
    def __init__(self, original_game_state: GameState):
        self.original_game_state = original_game_state

    def calculate_scores(self):
        return self.original_game_state.calculate_scores()

    def get_last_move(self):
        return ReversiMove(self.original_game_state.last_move)

    def get_next_game_states(self):
        return [ReversiGameState(next_state) for next_state in self.original_game_state.get_next_possible_moves()]

    def get_next_player(self):
        return self.original_game_state.calculate_next_player()

    def execute_move(self, move: ReversiMove):
        player, pos, choice = move.internal_tuple
        return ReversiGameState(self.original_game_state.execute_move(player, pos, choice))

    def get_player_list(self):
        return list(self.original_game_state.start_players)

    def wrap_in_evaluation(self):
        return ReversiEvaluation(self)

    def get_virtual_loss(self):
        return -0.5


class ReversiEvaluation(hometrainer.core.Evaluation):
    def __init__(self, game_state: ReversiGameState):
        super().__init__(game_state)

        # Add dummy values for move probabilities and expected game results
        next_game_states = self.game_state.get_next_game_states()
        tmp_possible_moves = [game_state.get_last_move() for game_state in next_game_states]
        self.probabilities = dict()
        for move in tmp_possible_moves:
            self.probabilities[move] = 0.0
        self.expected_result = dict()
        for player in self.game_state.get_player_list():
            self.expected_result[player] = 0.0

        # Keep it to be able to transform to/from normal form
        self.active_player = self.game_state.get_next_player()

        # Keep the applied transformations (rotation, mirroring) to be able to undo them.
        # TODO: write about dihedral groups in the thesis
        self._applied_transformations = []

        # Used to embedded smaller boards in big neural networks.
        # The top left position of where the board was inserted
        # has to be stored to reconstruct the outputs after the run.
        self.embedding_position = (0, 0)

    def get_expected_result(self):
        return self.expected_result

    def set_expected_result(self, expected_results):
        self.expected_result = expected_results

    def get_move_probabilities(self):
        return self.probabilities

    def set_move_probabilities(self, move_probabilities):
        self.probabilities = move_probabilities

    def convert_to_normal(self):
        current_active_player = self.game_state.get_next_player()

        # Active Player is already player one
        if current_active_player == Field.PLAYER_ONE:
            return self

        rotation_amount = (self.game_state.original_game_state.board.n_players -
                           Field.to_player_int(current_active_player) + 1)
        return self._rotate_players(rotation_amount)

    def convert_from_normal(self):
        current_active_player = self.game_state.get_next_player()

        # Active Player is already player one
        if current_active_player == self.active_player:
            return self

        rotation_amount = Field.to_player_int(self.active_player) - 1
        return self._rotate_players(rotation_amount)

    def get_possible_transformations(self):
        return range(0, 7)  # We allow 7 distinct transformations

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
        result = self

        for transformation in reversed(self._applied_transformations):
            result = result._undo_transformation(transformation)

        return result

    def _rotate_players(self, rotation_amount):
        """Rotate player numbers. Rotation by one means player one will be player two, and so on."""
        # Do not mutate our object. This could easily lead to hard to spot bugs.
        result = hometrainer.util.deepcopy(self)

        # Rotate the pieces on the board
        # TODO: Rotate in one go, not in multiple inversions. This is important for more than two players
        for i in range(rotation_amount):
            result.game_state.original_game_state.board.execute_inversion()

        n_players = self.game_state.original_game_state.board.n_players
        # Adjust values describing the current game state
        if self.game_state.original_game_state.last_move:
            player, pos, choice = self.game_state.original_game_state.last_move
            if isinstance(choice, Field):
                choice = Field.rotate_by(choice, rotation_amount, n_players)
            result.game_state.original_game_state.last_move = \
                (Field.rotate_by(player, rotation_amount, n_players), pos, choice)

        for player, bombs in self.game_state.original_game_state.player_bombs.items():
            result.game_state.original_game_state.player_bombs[Field.rotate_by(player, rotation_amount, n_players)] = bombs

        for player, overwrites in self.game_state.original_game_state.player_overwrites.items():
            result.game_state.original_game_state.player_overwrites[Field.rotate_by(player, rotation_amount, n_players)] = overwrites

        result.game_state.original_game_state.players = set()
        for player in self.game_state.original_game_state.players:
            result.game_state.original_game_state.players.add(Field.rotate_by(player, rotation_amount, n_players))

        if self.game_state.original_game_state._cached_next_player:
            result.game_state.original_game_state._cached_next_player = \
                Field.rotate_by(self.game_state.original_game_state._cached_next_player, rotation_amount, n_players)

        # Rotate evaluation stats
        result.expected_result = dict()
        for player, expected in self.expected_result.items():
            result.expected_result[Field.rotate_by(player, rotation_amount, n_players)] = expected

        result.probabilities = dict()
        for move, probability in self.probabilities.items():
            (player, pos, choice) = move.internal_tuple
            new_move = ReversiMove((Field.rotate_by(player, rotation_amount, n_players), pos, choice))
            result.probabilities[new_move] = probability

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
        height = self.game_state.original_game_state.board.height
        width = self.game_state.original_game_state.board.width

        # Currently only supported for quadratic boards
        assert (height == width)

        result = hometrainer.util.deepcopy(self)

        for y in range(height):
            for x in range(width):
                # Board
                new_pos = swap_function(x, y, width, height)
                result.game_state.original_game_state.board[new_pos] = self.game_state.original_game_state.board[(x, y)]

        # Move Probabilities
        result.probabilities = dict()
        for move, probability in self.probabilities.items():
            (player, (x, y), choice) = move.internal_tuple
            new_move = ReversiMove((player, swap_function(x, y, width, height), choice))
            result.probabilities[new_move] = probability

        # Last Move
        if self.game_state.original_game_state.last_move:
            last_player, (last_x, last_y), last_choice = self.game_state.original_game_state.last_move
            result.game_state.original_game_state.last_move = \
                last_player, swap_function(last_x, last_y, width, height), last_choice

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
