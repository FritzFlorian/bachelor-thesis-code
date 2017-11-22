"""ReversiXT Core Mechanics - Parse Board, Execute Move, Print Board"""
from enum import Enum, IntEnum
import io
import reversi.copy as copy
import itertools

class DisqualifiedError(Exception):
    """All 'known' client errors and timeouts lead to an disqualified exception for that player."""
    def __init__(self, player, message, cause=None):
        self.player = player
        self.message = message
        self.cause = cause


class GameState:
    """Fully describes a specific state of the game.

    The full state includes:
    - Last Move
    - Disqualified Players
    - Bomb Counts
    - Overwrite Counts
    - Current Board

    This means that each unique game state can be fully described by an GameState object.
    Board does not fulfill this, as it only captures the current 'visual' state of the board,
    ignoring prior actions by individual players.
    """

    def __init__(self, board):
        self.board = board
        self.last_move = None
        self._cached_next_player = None
        self.bomb_phase = False

        self.player_bombs = dict()
        self.player_overwrites = dict()
        self.players = set()
        for i in range(1, self.board.n_players + 1):
            player = i
            self.player_bombs[player] = board.n_bombs
            self.player_overwrites[player] = board.n_overwrite
            self.players.add(player)
        self.start_players = copy.copy(self.players)

    def execute_move(self, player, pos, choice):
        if not self.bomb_phase:
            possible = self.get_possible_moves_on_position(pos, player, True)
            for p in possible:
                if p.last_move == (player, pos, choice):
                    self._cached_next_player = player
                    return p

        possible = self.get_possible_bomb_move_on_position(pos, player)
        for p in possible:
            if p.last_move == (player, pos, choice):
                self._cached_next_player = player
                return p

        return None

    def get_possible_moves_on_position(self, pos, player=None, use_overwrite=False, reuse_list=None):
        if not player:
            player = self.next_player()

        if not reuse_list:
            reuse_list = []

        x, y = pos
        possible_moves = reuse_list
        type, new_board = \
            self.board.execute_move((x, y), player, use_overwrite and self.player_overwrites[player] > 0)

        # Handle different special move cases. Not really a good way to do them shorter.
        # Note that we do a lot of double copying here, this can be optimized, but is fine for now.
        if type == 'error':
            pass
        elif type == 'normal':
            new_game_state = self._minimal_copy(new_board, (player, (x, y), None))

            possible_moves.append(new_game_state)
        elif type == 'overwrite':
            new_game_state = self._minimal_copy(new_board, (player, (x, y), None))

            new_game_state.player_overwrites[player] = new_game_state.player_overwrites[player] - 1
            possible_moves.append(new_game_state)
        elif type == 'bonus':
            new_game_state = self._minimal_copy(new_board, (player, (x, y), 'bomb'))
            new_game_state_two = self._minimal_copy(new_board, (player, (x, y), 'overwrite'))

            new_game_state.player_bombs[player] = new_game_state.player_bombs[player] + 1
            new_game_state_two.player_overwrites[player] = new_game_state_two.player_overwrites[player] + 1

            possible_moves.append(new_game_state)
            possible_moves.append(new_game_state_two)
        elif type == 'choice':
            # Sorting is just to make this deterministic in tests
            for chosen_player in sorted(self.players):
                new_game_state = self._minimal_copy(copy.deepcopy(new_board), (player, (x, y), chosen_player))
                new_game_state.board.swap_stones(player, chosen_player)

                possible_moves.append(new_game_state)

        return possible_moves

    def _minimal_copy(self, new_board, last_move):
        new_game_state = copy.copy(self)
        new_game_state.player_bombs = copy.deepcopy(self.player_bombs)
        new_game_state.player_overwrites = copy.deepcopy(self.player_overwrites)

        new_game_state._cached_next_player = None
        new_game_state.board = new_board
        new_game_state.last_move = last_move

        return new_game_state

    def get_possible_bomb_move_on_position(self, pos, player=None):
        if not player:
            player = self.next_player()

        if self.player_bombs[player] <= 0:
            return []
        if self.board[pos] == Field.HOLE:
            return []

        new_game_state = copy.deepcopy(self)
        new_game_state._cached_next_player = None
        new_game_state.player_bombs[player] = self.player_bombs[player] - 1

        new_game_state.board.execute_bomb_at(self.board, pos)
        new_game_state.last_move = (player, pos, None)
        return [new_game_state]

    def get_possible_moves_for_player(self, player=None, use_overwrite=False):
        """Gets the possible moves for the next player only. Can be empty result if the next player can not move."""
        if not player:
            player = self.next_player()

        possible_moves = []
        if not self.bomb_phase:
            for x in range(self.board.width):
                for y in range(self.board.height):
                    possible_moves = self.get_possible_moves_on_position((x, y), player=player, use_overwrite=use_overwrite, reuse_list=possible_moves)

        return possible_moves

    def get_possible_bomb_moves_for_player(self, player=None):
        """Gets the possible bomb moves for the next player only.
        Can be empty result if the next player can not move."""
        if not self.bomb_phase:
            return []

        if not player:
            player = self.next_player()

        possible_moves = []
        for x in range(self.board.width):
            for y in range(self.board.height):
                possible_moves.append(
                    self.get_possible_bomb_move_on_position((x, y), player=player))

        return list(itertools.chain.from_iterable(possible_moves))

    def get_next_possible_moves(self):
        """Finds the next list of moves that can be executed with regard to the player ordering.
        :return: A list of possible follow up game states
        """
        next_player = self.next_player()

        if not self.bomb_phase:
            for i in range(self.board.n_players):
                possible_moves = self.get_possible_moves_for_player(player=next_player, use_overwrite=True)
                if len(possible_moves) > 0:
                    self._cached_next_player = possible_moves[0].last_move[0]
                    return possible_moves

                next_player = self.next_player(next_player)

        self.bomb_phase = True
        next_player = self.next_player()
        for i in range(self.board.n_players):
            possible_moves = self.get_possible_bomb_moves_for_player(player=next_player)
            if len(possible_moves) > 0:
                for move in possible_moves:
                    move._cached_next_player = None
                self._cached_next_player = possible_moves[0].last_move[0]
                return possible_moves

            next_player = self.next_player(next_player)

        self._cached_next_player = None
        return []

    def next_player(self, player=None):
        """Finds the next player according to the player ordering and disqualified players.

        Note: This ignores if the next player can actually make a move.
        :param player: Optional player. Will then ignore the last move and use this as the last player.
        :return: The next player to try out a move for or None if all are disqualified.
        """
        # look up the last player to find the next one
        if not player:
            if not self.last_move:
                return Field.PLAYER_ONE
            (player, _, _) = self.last_move

        # Find the next, not disqualified player
        raw_player = player - Field.PLAYER_ONE
        for i in range(1, self.board.n_players + 1):
            new_raw_player = (raw_player + i) % self.board.n_players
            new_player = new_raw_player + Field.PLAYER_ONE
            if new_player in self.players:
                return new_player

        return None

    def calculate_next_player(self):
        """Finds the next player according to the player order AND possible moves.

        This will actually find the next possible moves and derive the next player that can move.
        (This is needed in reversi as there can be situations where one player can not move and
        is skipped.)"""
        if self._cached_next_player is not None:
            return self._cached_next_player

        next_moves = self.get_next_possible_moves()
        if len(next_moves) > 0:
            self._cached_next_player = next_moves[0].last_move[0]
            return next_moves[0].last_move[0]

        return None

    def disqualify_player(self, player):
        self.players.remove(player)

    def calculate_scores(self):
        stone_counts = self.board.count(self.players).items()

        # Start scores as -1 in case anyone got disqualified
        scores = {player: -1.0 for player in self.start_players}

        # Get the ranks of all players according to ReversiXT tournament rules
        ranks = self._rank_from_stone_counts(stone_counts)

        # IMPORTANT
        # We give different scores then reversi.
        # Usually tied players all get the higher score of the top rank that they tide in.
        # We MUST NOT do this, because this would destroy the zero sum property of our game
        # and the neural network might learn to always play perfect ties, as this leads to the
        # biggest expected value for everyone (and as it's trained by selfplay, it can easily
        # achieve this, as every client is on page with this perfect game plan)
        # TODO: Write about this in the thesis, it's an important point of game theory
        for rank, players in ranks.items():
            if len(players) == 0:
                continue

            total_score = 0
            for i in range(0, len(players)):
                total_score = total_score + self._score_for_rank(rank + i)
            score = total_score / len(players)

            for player in players:
                scores[player] = score

        return scores

    def _rank_from_stone_counts(self, stone_counts):
        sorted_counts = sorted(stone_counts, key=lambda entry: entry[1], reverse=True)

        current_rank = 0
        current_iteration = 0
        last_stone_count = -1
        ranks = {rank: [] for rank in range(1, 9)}
        for player, stone_count in sorted_counts:
            current_iteration = current_iteration + 1
            if last_stone_count != stone_count:
                current_rank = current_iteration
                last_stone_count = stone_count

            ranks[current_rank].append(player)

        return ranks

    def _score_for_rank(self, rank):
        # TODO: Add better scoring for more than two players
        if rank == 1:
            return 1.0
        if rank == 2:
            return -1.0

        return -1.0


class Board:
    """The current assignment and shape of the game board.

    This class holds the current game board as well as transitions and
    various metadata about the game board.
    Does not hold information on the current state of individual players
    (e.g. disqualifications, bomb counts, next active player, ...)."""

    def __init__(self, board_string):
        """Parses a map string into a Board.

        :param board_string: map encoded as a string
        """
        self._board = None
        self.transitions = dict()

        buf = io.StringIO(board_string)

        # Read Header
        self.n_players = int(buf.readline())
        self.n_overwrite = int(buf.readline())
        bomb_info = buf.readline().split()
        self.n_bombs, self.s_bombs = int(bomb_info[0]), int(bomb_info[1])
        dimension_info = buf.readline().split()
        self.height, self.width = int(dimension_info[0]), int(dimension_info[1])

        self.read_board(buf)
        self.read_transitions(buf)

    def read_board(self, buf):
        self._board = []

        for y in range(self.height):
            input_row = buf.readline().split()
            output_row = []
            for x in range(self.width):
                output_row.append(FIELD_LOOKUP.index(input_row[x]) - 1)
            self._board.append(output_row)

    def read_transitions(self, buf):
        for transition in buf:
            components = transition.split()
            if len(components) < 7:
                return

            start_pos = (int(components[0]), int(components[1]))
            start_dir = Direction(int(components[2]))
            end_pos = (int(components[4]), int(components[5]))
            end_dir = Direction(int(components[6]))

            self.transitions[(start_pos, start_dir)] = (end_pos, Direction.mirror(end_dir))
            self.transitions[(end_pos, end_dir)] = (start_pos, Direction.mirror(start_dir))

    def execute_move(self, pos, player, use_overwrite=True):
        if pos not in self:
            return 'error', None

        # Start with checks if the position is occupied
        if self[pos] == Field.HOLE:
            return 'error', None

        need_overwrite = False
        if self.is_player_at(pos) or self[pos] == Field.EXPANSION:
            need_overwrite = True

        if need_overwrite and not use_overwrite:
            return 'error', None

        captured_positions = [pos]
        # Don't question this, it makes python way faster
        _next_pos = self._next_pos

        for dir in Direction:
            cur_captured = []
            # Don't question this, it makes python way faster
            append = cur_captured.append
            cur_pos, cur_dir = _next_pos(pos, dir)

            while True:
                cur_x, cur_y = cur_pos
                if cur_x < 0 or cur_x >= self.width or cur_y < 0 or cur_y >= self.height:
                    break

                cur_field = self._board[cur_y][cur_x]

                if not ((cur_field in PLAYERS or cur_field == Field.EXPANSION) and cur_field != player and cur_pos != pos):
                    break
                append(cur_pos)
                cur_pos, cur_dir = _next_pos(cur_pos, cur_dir)

            # Need to end a line at our player
            if cur_pos in self and self[cur_pos] == player and cur_pos != pos:
                captured_positions = captured_positions + cur_captured

        if len(captured_positions) <= 1 and self[pos] != Field.EXPANSION:
            return 'error', None

        next_board = copy.deepcopy(self)
        for capt_pos in captured_positions:
            next_board[capt_pos] = player

        if need_overwrite:
            return 'overwrite', next_board
        if self[pos] == Field.CHOICE:
            return 'choice', next_board
        if self[pos] == Field.BONUS:
            return 'bonus', next_board

        if self[pos] == Field.INVERSION:
            next_board.execute_inversion()
        return 'normal', next_board

    def execute_inversion(self):
        for x in range(self.width):
            for y in range(self.height):
                if self.is_player_at((x, y)):
                    raw_value = self[(x, y)] - Field.PLAYER_ONE
                    raw_value = (raw_value + 1) % self.n_players
                    self[(x, y)] = raw_value + Field.PLAYER_ONE

    def swap_stones(self, first, second):
        for x in range(self.width):
            for y in range(self.height):
                if self[(x, y)] == first:
                    self[(x, y)] = second
                elif self[(x, y)] == second:
                    self[(x, y)] = first

    def execute_bomb_at(self, old_board, pos):
        self._execute_bomb_at_recursive(old_board, pos, self.s_bombs)

    def _execute_bomb_at_recursive(self, old_board, pos, strength):
        # Stop at walls an if the strength runs out
        if strength < 0:
            return
        if pos not in old_board or old_board[pos] == Field.HOLE:
            return

        self[pos] = Field.HOLE
        for dir in Direction:
            new_pos, _ = self._next_pos(pos, dir)
            self._execute_bomb_at_recursive(old_board, new_pos, strength - 1)

    def _next_pos(self, pos, dir):
        if (pos, dir) in self.transitions:
            return self.transitions[(pos, dir)]

        x, y = pos
        x_add, y_add = MOVEMENT[dir]
        return (x + x_add, y + y_add), dir

    def is_player_at(self, pos):
        return pos in self and self[pos] in PLAYERS

    def count(self, fields):
        """Count the number of individual fields on the board. Counts only the values given in fields."""
        result = {field: 0 for field in fields}
        for x in range(0, self.width):
            for y in range(0, self.height):
                tmp = self[(x, y)]
                if tmp in fields:
                    result[tmp] = result[tmp] + 1

        return result

    def __string__(self):
        str_list = list()

        str_list.append(str(self.n_players))
        str_list.append(str(self.n_overwrite))
        str_list.append("{} {}".format(self.n_bombs, self.s_bombs))
        str_list.append("{} {}".format(self.height, self.width))
        str_list.append(self.board_string())

        for k, v in self.transitions.items():
            (x_1, y_1), dir_1 = k
            (x_2, y_2), dir_2 = v
            str_list.append("{} {} {} <-> {} {} {}"
                            .format(x_1, y_1, dir_1.value, x_2, y_2, Direction.mirror(dir_2).value))

        return '\n'.join(str_list)

    def board_string(self):
        str_list = list()

        for y in range(self.height):
            line_items = []
            for x in range(self.width):
                line_items.append(Field.to_string(self[(x, y)]))
            str_list.append(" ".join(line_items))

        return '\n'.join(str_list)

    def __getitem__(self, key):
        """Allows simple access to the fields of the board.

        Converts the internal int8 representation to the Field enums.
        This allows to work with the board at a high level, without knowing any implementation details."""
        x, y = key
        if x < 0 or x >= self.width:
            raise KeyError('x not in board bounds!')
        if y < 0 or y >= self.height:
            raise KeyError('y not in board bounds!')

        return self._board[y][x]

    def __setitem__(self, key, value):
        """Allows simple set operations on the board.

        Converts the Field enums to the internal int8 representation.
        This allows to work with the board at a high level, without knowing any implementation details."""
        x, y = key
        if x < 0 or x >= self.width:
            raise KeyError('x not in board bounds!')
        if y < 0 or y >= self.height:
            raise KeyError('y not in board bounds!')

        self._board[y][x] = value

    def __contains__(self, key):
        x, y = key
        if x < 0 or x >= self.width:
            return False
        if y < 0 or y >= self.height:
            return False

        return True


class Field:
    HOLE = -1
    EMPTY = 0

    PLAYER_ONE = 1
    PLAYER_TWO = 2
    PLAYER_THREE = 3
    PLAYER_FOUR = 4
    PLAYER_FIVE = 5
    PLAYER_SIX = 6
    PLAYER_SEVEN = 7
    PLAYER_EIGHT = 8

    INVERSION = 9
    CHOICE = 10
    EXPANSION = 11
    BONUS = 12

    # Transformations useful when rotating the board
    @staticmethod
    def rotate_by(field, amount, n_players):
        new_player_number = ((Field.to_player_int(field) - 1 + amount) % n_players) + 1
        return Field.int_to_player(new_player_number)

    @staticmethod
    def to_player_int(field):
        return field - Field.PLAYER_ONE + 1

    @staticmethod
    def int_to_player(number):
        return number + Field.PLAYER_ONE - 1

    @staticmethod
    def to_string(field):
        return FIELD_LOOKUP[field + 1]


PLAYERS = {Field.PLAYER_ONE, Field.PLAYER_TWO, Field.PLAYER_THREE, Field.PLAYER_FOUR,
           Field.PLAYER_FIVE, Field.PLAYER_SIX, Field.PLAYER_SEVEN, Field.PLAYER_EIGHT}


class Direction(Enum):
    TOP = 0
    TOP_RIGHT = 1
    RIGHT = 2
    BOTTOM_RIGHT = 3
    BOTTOM = 4
    BOTTOM_LEFT = 5
    LEFT = 6
    TOP_LEFT = 7

    @staticmethod
    def mirror(direction):
        new_val = (direction.value + 4) % 8
        return Direction(new_val)


MOVEMENT = {
    Direction.TOP: (0, -1),
    Direction.TOP_RIGHT: (1, -1),
    Direction.RIGHT: (1, 0),
    Direction.BOTTOM_RIGHT: (1, 1),
    Direction.BOTTOM: (0, 1),
    Direction.BOTTOM_LEFT: (-1, 1),
    Direction.LEFT: (-1, 0),
    Direction.TOP_LEFT: (-1, -1)
}

FIELD_LOOKUP = ('-', '0', '1', '2', '3', '4', '5', '6', '7', '8', 'i', 'c', 'x', 'b')
