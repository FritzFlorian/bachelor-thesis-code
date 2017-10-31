import unittest
from reversi.game_core import Board, Field, Direction


class TestBoardParsing(unittest.TestCase):
    """Parsing & other input/output tests
    """

    simple_board = """\
    2
    3
    4 5
    2 4
    - 0 1 2
    x c i b
    """

    transition_board = """\
    2
    0
    0 0
    2 2
    0 0
    0 0
    0 0 7 <-> 1 0 1
    0 1 6 <-> 0 1 4
    """

    def test_parse_basic_board(self):
        board = Board(TestBoardParsing.simple_board)

        self.assertEqual(board.n_players, 2)
        self.assertEqual(board.n_overwrite, 3)
        self.assertEqual(board.n_bombs, 4)
        self.assertEqual(board.s_bombs, 5)
        self.assertEqual(board.height, 2)
        self.assertEqual(board.width, 4)

        self.assertEqual(board.board[(0, 0)], Field.HOLE)
        self.assertEqual(board.board[(1, 0)], Field.EMPTY)
        self.assertEqual(board.board[(2, 0)], Field.PLAYER_ONE)
        self.assertEqual(board.board[(3, 0)], Field.PLAYER_TWO)
        self.assertEqual(board.board[(0, 1)], Field.EXPANSION)
        self.assertEqual(board.board[(1, 1)], Field.CHOICE)
        self.assertEqual(board.board[(2, 1)], Field.INVERSION)
        self.assertEqual(board.board[(3, 1)], Field.BONUS)

    def test_parse_transitions(self):
        board = Board(TestBoardParsing.transition_board)

        self.assertEqual(board.transitions[((0, 0), Direction.TOP_LEFT)], ((1, 0), Direction.TOP_RIGHT))
        self.assertEqual(board.transitions[((1, 0), Direction.TOP_RIGHT)], ((0, 0), Direction.TOP_LEFT))

        self.assertEqual(board.transitions[((0, 1), Direction.LEFT)], ((0, 1), Direction.BOTTOM))
        self.assertEqual(board.transitions[((0, 1), Direction.BOTTOM)], ((0, 1), Direction.LEFT))


if __name__ == '__main__':
    unittest.main()
