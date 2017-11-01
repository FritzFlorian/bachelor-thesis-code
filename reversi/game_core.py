from enum import Enum
import io
import copy


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

        self.player_bombs = dict()
        self.player_overwrites = dict()
        self.players = set()
        for i in range(self.board.n_players):
            player = Field(chr(i + ord(Field.PLAYER_ONE.value)))
            self.player_bombs[player] = board.n_bombs
            self.player_overwrites[player] = board.n_overwrite
            self.players.add(player)

    def get_possible_moves_for_player(self, player=None, use_overwrite=False):
        """Gets the possible moves for the next player only. Can be empty result if the next player can not move."""
        if not player:
            player = self.next_player()

        possible_moves = []
        for x in range(self.board.width):
            for y in range(self.board.height):
                type, new_board = \
                    self.board.execute_move((x, y), player, use_overwrite and self.player_overwrites[player] > 0)

                # Handle different special move cases. Not really a good way to do them shorter.
                # Note that we do a lot of double copying here, this can be optimized, but is fine for now.
                if type == 'error':
                    continue
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
                    new_game_state.player_bombs = new_game_state.player_bombs[player] + 1

                    new_game_state_two.last_move = (player, (x, y), 'overwrite')
                    new_game_state_two.player_overwrites[player] = new_game_state_two.player_overwrites[player] +1

                    possible_moves.append(new_game_state)
                    possible_moves.append(new_game_state_two)
                elif type == 'choice':
                    for chosen_player in self.players:
                        new_game_state = copy.deepcopy(self)
                        new_game_state.board = copy.deepcopy(new_board)
                        new_game_state.board.swap_stones(player, chosen_player)
                        new_game_state.last_move = (player, (x, y), chosen_player)

                        possible_moves.append(new_game_state)

        return possible_moves

    def get_next_possible_moves(self):
        """Finds the next list of moves that can be executed with regard to the player ordering.
        :return: A list of possible follow up game states
        """
        next_player = self.next_player()
        for i in range(self.board.n_players):
            possible_moves = self.get_possible_moves_for_player(player=next_player, use_overwrite=True)
            if len(possible_moves) > 0:
                return possible_moves

        return None

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

            self.transitions[(start_pos, start_dir)] = (end_pos, end_dir)
            self.transitions[(end_pos, end_dir)] = (start_pos, start_dir)

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
            while cur_pos in self.board and self._is_walkable(cur_pos, player):
                cur_captured.append(cur_pos)
                cur_pos, cur_dir = self._next_pos(cur_pos, cur_dir)

            # Need to end a line at our player
            if cur_pos in self.board and self.board[cur_pos] == player:
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
            self.execute_inversion()
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

    def __string__(self):
        str_list = list()

        str_list.append(str(self.n_players))
        str_list.append(str(self.n_overwrite))
        str_list.append("{} {}".format(self.n_bombs, self.s_bombs))
        str_list.append("{} {}".format(self.height, self.width))
        for y in range(self.height):
            line_items = []
            for x in range(self.width):
                line_items.append(self.board[(x, y)].value)
            str_list.append(" ".join(line_items))

        for k, v in self.transitions.items():
            str_list.append("{} <-> {}".format(k, v))

        return '\n'.join(str_list)

    def __getitem__(self, arg):
        return self.board[arg]


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
