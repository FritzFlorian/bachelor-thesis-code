from enum import Enum
import io


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
        self.board = None


class Board:
    def __init__(self, board_string):
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


class Direction(Enum):
    TOP = 0
    TOP_RIGHT = 1
    RIGHT = 2
    BOTTOM_RIGHT = 3
    BOTTOM = 4
    BOTTOM_LEFT = 5
    LEFT = 6
    TOP_LEFT = 7
