"""ReversiXT Core Mechanics - Parse Board, Execute Move, Print Board"""
from enum import Enum
import io
import copy
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
        for i in range(self.board.n_players):
            player = Field(chr(i + ord(Field.PLAYER_ONE.value)))
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

    def get_possible_moves_on_position(self, pos, player=None, use_overwrite=False):
        if not player:
            player = self.next_player()

        x, y = pos
        possible_moves = []
        type, new_board = \
            self.board.execute_move((x, y), player, use_overwrite and self.player_overwrites[player] > 0)

        # Handle different special move cases. Not really a good way to do them shorter.
        # Note that we do a lot of double copying here, this can be optimized, but is fine for now.
        if type == 'error':
            pass
        elif type == 'normal':
            new_game_state = copy.deepcopy(self)
            new_game_state.board = new_board
            new_game_state.last_move = (player, (x, y), None)

            possible_moves.append(new_game_state)
        elif type == 'overwrite':
            new_game_state = copy.deepcopy(self)
            new_game_state.board = new_board
            new_game_state.last_move = (player, (x, y), None)

            new_game_state.player_overwrites[player] = new_game_state.player_overwrites[player] - 1
            possible_moves.append(new_game_state)
        elif type == 'bonus':
            new_game_state = copy.deepcopy(self)
            new_game_state.board = new_board
            new_game_state_two = copy.deepcopy(new_game_state)

            new_game_state.last_move = (player, (x, y), 'bomb')
            new_game_state.player_bombs[player] = new_game_state.player_bombs[player] + 1

            new_game_state_two.last_move = (player, (x, y), 'overwrite')
            new_game_state_two.player_overwrites[player] = new_game_state_two.player_overwrites[player] + 1

            possible_moves.append(new_game_state)
            possible_moves.append(new_game_state_two)
        elif type == 'choice':
            # Sorting is just to make this deterministic in tests
            for chosen_player in sorted(self.players):
                new_game_state = copy.deepcopy(self)
                new_game_state.board = copy.deepcopy(new_board)
                new_game_state.board.swap_stones(player, chosen_player)
                new_game_state.last_move = (player, (x, y), chosen_player)

                possible_moves.append(new_game_state)

        for move in possible_moves:
            move._cached_next_player = None

        return possible_moves

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
                    possible_moves.append(self.get_possible_moves_on_position((x, y), player=player, use_overwrite=use_overwrite))

        return list(itertools.chain.from_iterable(possible_moves))

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
                    for move in possible_moves:
                        move._cached_next_player = None
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
        raw_player = ord(player.value) - ord(Field.PLAYER_ONE.value)
        for i in range(1, self.board.n_players + 1):
            new_raw_player = (raw_player + i) % self.board.n_players
            new_player = Field(chr(new_raw_player + ord(Field.PLAYER_ONE.value)))
            if new_player in self.players:
                return new_player

        return None

    def calculate_next_player(self):
        """Finds the next player according to the player order AND possible moves.

        This will actually find the next possible moves and derive the next player that can move.
        (This is needed in reversi as there can be situations where one player can not move and
        is skipped.)"""
        if self._cached_next_player:
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
            return 0.0


        return 0.0


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
        self.board = dict()
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
        for y in range(self.height):
            row = buf.readline().split()
            for x in range(self.width):
                self.board[(x, y)] = Field(row[x])

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
        if pos not in self.board:
            return 'error', None

        # Start with checks if the position is occupied
        if self.board[pos] == Field.HOLE:
            return 'error', None

        need_overwrite = False
        if self.is_player_at(pos) or self.board[pos] == Field.EXPANSION:
            need_overwrite = True

        if need_overwrite and not use_overwrite:
            return 'error', None

        captured_positions = [pos]
        for dir in Direction:
            cur_captured = []
            cur_pos, cur_dir = self._next_pos(pos, dir)
            while cur_pos in self.board and self._is_walkable(cur_pos, player) and cur_pos != pos:
                cur_captured.append(cur_pos)
                cur_pos, cur_dir = self._next_pos(cur_pos, cur_dir)

            # Need to end a line at our player
            if cur_pos in self.board and self.board[cur_pos] == player and cur_pos != pos:
                captured_positions = captured_positions + cur_captured

        if len(captured_positions) <= 1 and self.board[pos] != Field.EXPANSION:
            return 'error', None

        next_board = copy.deepcopy(self)
        for capt_pos in captured_positions:
            next_board.board[capt_pos] = player

        if need_overwrite:
            return 'overwrite', next_board
        if self.board[pos] == Field.CHOICE:
            return 'choice', next_board
        if self.board[pos] == Field.BONUS:
            return 'bonus', next_board

        if self.board[pos] == Field.INVERSION:
            next_board.execute_inversion()
        return 'normal', next_board

    def execute_inversion(self):
        for x in range(self.width):
            for y in range(self.height):
                if self.is_player_at((x, y)):
                    raw_value = ord(self.board[(x, y)].value) - ord(Field.PLAYER_ONE.value)
                    raw_value = (raw_value + 1) % self.n_players
                    self.board[(x, y)] = Field(chr(raw_value + ord(Field.PLAYER_ONE.value)))

    def swap_stones(self, first, second):
        for x in range(self.width):
            for y in range(self.height):
                if self.board[(x, y)] == first:
                    self.board[(x, y)] = second
                elif self.board[(x, y)] == second:
                    self.board[(x, y)] = first

    def execute_bomb_at(self, old_board, pos):
        self._execute_bomb_at_recursive(old_board, pos, self.s_bombs)

    def _execute_bomb_at_recursive(self, old_board, pos, strength):
        # Stop at walls an if the strength runs out
        if strength < 0:
            return
        if old_board[pos] == Field.HOLE or not old_board[pos]:
            return

        self.board[pos] = Field.HOLE
        for dir in Direction:
            new_pos, _ = self._next_pos(pos, dir)
            self._execute_bomb_at_recursive(old_board, new_pos, strength - 1)

    def _is_walkable(self, pos, player):
        return (self.is_player_at(pos) or self.board[pos] == Field.EXPANSION) and self.board[pos] != player

    def _next_pos(self, pos, dir):
        if (pos, dir) in self.transitions:
            return self.transitions[(pos, dir)]

        x, y = pos
        if dir == Direction.TOP:
            return (x, y - 1), dir
        if dir == Direction.TOP_RIGHT:
            return (x + 1, y - 1), dir
        if dir == Direction.RIGHT:
            return (x + 1, y), dir
        if dir == Direction.BOTTOM_RIGHT:
            return (x + 1, y + 1), dir
        if dir == Direction.BOTTOM:
            return (x, y + 1), dir
        if dir == Direction.BOTTOM_LEFT:
            return (x - 1, y + 1), dir
        if dir == Direction.LEFT:
            return (x - 1, y), dir
        if dir == Direction.TOP_LEFT:
            return (x - 1, y - 1), dir

    def is_player_at(self, pos):
        return pos in self.board and self.board[pos] in PLAYERS

    def count(self, fields):
        """Count the number of individual fields on the board. Counts only the values given in fields."""
        result = {field: 0 for field in fields}
        for x in range(0, self.width):
            for y in range(0, self.height):
                tmp = self.board[(x, y)]
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
                line_items.append(self.board[(x, y)].value)
            str_list.append(" ".join(line_items))

        return '\n'.join(str_list)

    def __getitem__(self, arg):
        if arg in self.board:
            return self.board[arg]
        return None


class Field(Enum):
    EMPTY = '0'

    PLAYER_ONE = '1'
    PLAYER_TWO = '2'
    PLAYER_THREE = '3'
    PLAYER_FOUR = '4'
    PLAYER_FIVE = '5'
    PLAYER_SIX = '6'
    PLAYER_SEVEN = '7'
    PLAYER_EIGHT = '8'

    INVERSION = 'i'
    CHOICE = 'c'
    EXPANSION = 'x'
    BONUS = 'b'

    HOLE = '-'

    def to_int(self):
        return ord(self.value) - ord(Field.PLAYER_ONE.value) + 1

    def rotate_by(self, amount, n_players):
        new_player_number = ((self.to_int() - 1 + amount) % n_players) + 1
        return Field.int_to_player(new_player_number)

    @staticmethod
    def int_to_player(number):
        return Field(chr(number + ord(Field.PLAYER_ONE.value) - 1))

    def __lt__(self, other):
        return ord(self.value) < ord(other.value)


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
