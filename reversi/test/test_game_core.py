import unittest
from reversi.game_core import Board, Field, Direction, GameState

HOLE = Field.HOLE
EMPTY = Field.EMPTY
PLAYER_ONE = Field.PLAYER_ONE
PLAYER_TWO = Field.PLAYER_TWO
PLAYER_THREE = Field.PLAYER_THREE
PLAYER_FOUR = Field.PLAYER_FOUR
PLAYER_FIVE = Field.PLAYER_FIVE
PLAYER_SIX = Field.PLAYER_SIX
PLAYER_SEVEN = Field.PLAYER_SEVEN
PLAYER_EIGHT = Field.PLAYER_EIGHT
EXPANSION = Field.EXPANSION
CHOICE = Field.CHOICE
INVERSION = Field.INVERSION
BONUS = Field.BONUS


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

        self.assertEqual(board[(0, 0)], HOLE)
        self.assertEqual(board[(1, 0)], EMPTY)
        self.assertEqual(board[(2, 0)], PLAYER_ONE)
        self.assertEqual(board[(3, 0)], PLAYER_TWO)
        self.assertEqual(board[(0, 1)], EXPANSION)
        self.assertEqual(board[(1, 1)], CHOICE)
        self.assertEqual(board[(2, 1)], INVERSION)
        self.assertEqual(board[(3, 1)], BONUS)

    def test_parse_transitions(self):
        board = Board(TestBoardParsing.transition_board)

        self.assertEqual(board.transitions[((0, 0), Direction.TOP_LEFT)], ((1, 0), Direction.BOTTOM_LEFT))
        self.assertEqual(board.transitions[((1, 0), Direction.TOP_RIGHT)], ((0, 0), Direction.BOTTOM_RIGHT))

        self.assertEqual(board.transitions[((0, 1), Direction.LEFT)], ((0, 1), Direction.TOP))
        self.assertEqual(board.transitions[((0, 1), Direction.BOTTOM)], ((0, 1), Direction.RIGHT))


class TestScoring(unittest.TestCase):
    def test_normal_scoring(self):
        board = Board("""\
        2
        1
        0 0
        3 3
        1 0 0
        0 0 0
        0 0 0
        """)
        game = GameState(board)
        scores = game.calculate_scores()

        self.assertEqual(scores[PLAYER_ONE], 1.0)
        self.assertEqual(scores[PLAYER_TWO], 0.0)

    def test_tie_scoring(self):
        board = Board("""\
        3
        1
        0 0
        3 3
        1 2 0
        0 0 0
        0 0 0
        """)
        game = GameState(board)
        scores = game.calculate_scores()

        self.assertEqual(scores[PLAYER_ONE], 0.5)
        self.assertEqual(scores[PLAYER_TWO], 0.5)
        self.assertEqual(scores[PLAYER_THREE], 0.0)


class TestMoveExecution(unittest.TestCase):
    def test_get_possible_moves_walk_bottom(self):
        board = Board("""\
        2
        0
        0 0
        3 3
        0 1 0
        0 2 0
        0 0 0
        """)
        game = GameState(board)

        next_game_states = game.get_possible_moves_for_player(PLAYER_ONE)
        self.assertEqual(1, len(next_game_states))
        self.assert_board(next_game_states[0].board,
                          [[EMPTY, PLAYER_ONE, EMPTY],
                           [EMPTY, PLAYER_ONE, EMPTY],
                           [EMPTY, PLAYER_ONE, EMPTY]])

    def test_get_possible_moves_walk_bottom_left(self):
        board = Board("""\
        2
        0
        0 0
        3 3
        0 0 1
        0 2 0
        0 0 0
        """)
        game = GameState(board)

        next_game_states = game.get_possible_moves_for_player(PLAYER_ONE)
        self.assertEqual(1, len(next_game_states))
        self.assert_board(next_game_states[0].board,
                          [[EMPTY, EMPTY, PLAYER_ONE],
                           [EMPTY, PLAYER_ONE, EMPTY],
                           [PLAYER_ONE, EMPTY, EMPTY]])

    def test_get_possible_moves_walk_left(self):
        board = Board("""\
        2
        0
        0 0
        3 3
        0 0 0
        0 2 1
        0 0 0
        """)
        game = GameState(board)

        next_game_states = game.get_possible_moves_for_player(PLAYER_ONE)
        self.assertEqual(1, len(next_game_states))
        self.assert_board(next_game_states[0].board,
                          [[EMPTY, EMPTY, EMPTY],
                           [PLAYER_ONE, PLAYER_ONE, PLAYER_ONE],
                           [EMPTY, EMPTY, EMPTY]])

    def test_get_possible_moves_walk_top_left(self):
        board = Board("""\
        2
        0
        0 0
        3 3
        0 0 0
        0 2 0
        0 0 1
        """)
        game = GameState(board)

        next_game_states = game.get_possible_moves_for_player(PLAYER_ONE)
        self.assertEqual(1, len(next_game_states))
        self.assert_board(next_game_states[0].board,
                          [[PLAYER_ONE, EMPTY, EMPTY],
                           [EMPTY, PLAYER_ONE, EMPTY],
                           [EMPTY, EMPTY, PLAYER_ONE]])

    def test_get_possible_moves_walk_top(self):
        board = Board("""\
        2
        0
        0 0
        3 3
        0 0 0
        0 2 0
        0 1 0
        """)
        game = GameState(board)

        next_game_states = game.get_possible_moves_for_player(PLAYER_ONE)
        self.assertEqual(1, len(next_game_states))
        self.assert_board(next_game_states[0].board,
                          [[EMPTY, PLAYER_ONE, EMPTY],
                           [EMPTY, PLAYER_ONE, EMPTY],
                           [EMPTY, PLAYER_ONE, EMPTY]])

    def test_get_possible_moves_walk_top_right(self):
        board = Board("""\
        2
        0
        0 0
        3 3
        0 0 0
        0 2 0
        1 0 0
        """)
        game = GameState(board)

        next_game_states = game.get_possible_moves_for_player(PLAYER_ONE)
        self.assertEqual(1, len(next_game_states))
        self.assert_board(next_game_states[0].board,
                          [[EMPTY, EMPTY, PLAYER_ONE],
                           [EMPTY, PLAYER_ONE, EMPTY],
                           [PLAYER_ONE, EMPTY, EMPTY]])

    def test_get_possible_moves_walk_right(self):
        board = Board("""\
        2
        0
        0 0
        3 3
        0 0 0
        1 2 0
        0 0 0
        """)
        game = GameState(board)

        next_game_states = game.get_possible_moves_for_player(PLAYER_ONE)
        self.assertEqual(1, len(next_game_states))
        self.assert_board(next_game_states[0].board,
                          [[EMPTY, EMPTY, EMPTY],
                           [PLAYER_ONE, PLAYER_ONE, PLAYER_ONE],
                           [EMPTY, EMPTY, EMPTY]])

    def test_get_possible_moves_walk_bottom_right(self):
        board = Board("""\
        2
        0
        0 0
        3 3
        1 0 0
        0 2 0
        0 0 0
        """)
        game = GameState(board)

        next_game_states = game.get_possible_moves_for_player(PLAYER_ONE)
        self.assertEqual(1, len(next_game_states))
        self.assert_board(next_game_states[0].board,
                          [[PLAYER_ONE, EMPTY, EMPTY],
                           [EMPTY, PLAYER_ONE, EMPTY],
                           [EMPTY, EMPTY, PLAYER_ONE]])

    def test_get_possible_moves_walk_multipath(self):
        board = Board("""\
        2
        0
        0 0
        3 3
        1 0 0
        0 2 2
        0 0 0
        """)
        game = GameState(board)

        next_game_states = game.get_possible_moves_for_player(PLAYER_ONE)
        self.assertEqual(1, len(next_game_states))
        self.assert_board(next_game_states[0].board,
                          [[PLAYER_ONE, EMPTY, EMPTY],
                           [EMPTY, PLAYER_ONE, PLAYER_TWO],
                           [EMPTY, EMPTY, PLAYER_ONE]])

    def test_get_possible_moves_expansion(self):
        board = Board("""\
        2
        1
        0 0
        3 3
        0 0 0
        0 x 0
        0 0 0
        """)
        game = GameState(board)

        next_game_states = game.get_possible_moves_for_player(PLAYER_ONE, use_overwrite=True)
        self.assertEqual(1, len(next_game_states))
        self.assert_board(next_game_states[0].board,
                          [[EMPTY, EMPTY, EMPTY],
                           [EMPTY, PLAYER_ONE, EMPTY],
                           [EMPTY, EMPTY, EMPTY]])
        self.assertEqual(0, next_game_states[0].player_overwrites[PLAYER_ONE])

    def test_get_possible_moves_capture_expansion_rows(self):
        board = Board("""\
        2
        0
        0 0
        3 3
        0 0 0
        0 x 0
        0 0 1
        """)
        game = GameState(board)

        next_game_states = game.get_possible_moves_for_player(PLAYER_ONE, use_overwrite=True)
        self.assertEqual(1, len(next_game_states))
        self.assert_board(next_game_states[0].board,
                          [[PLAYER_ONE, EMPTY, EMPTY],
                           [EMPTY, PLAYER_ONE, EMPTY],
                           [EMPTY, EMPTY, PLAYER_ONE]])
        self.assertEqual(0, next_game_states[0].player_overwrites[PLAYER_ONE])

    def test_get_possible_moves_overwrite_on_other_player(self):
        board = Board("""\
        2
        1
        0 0
        3 3
        2 0 0
        0 2 0
        0 0 1
        """)
        game = GameState(board)

        next_game_states = game.get_possible_moves_for_player(PLAYER_ONE, use_overwrite=True)
        self.assertEqual(1, len(next_game_states))
        self.assert_board(next_game_states[0].board,
                          [[PLAYER_ONE, EMPTY, EMPTY],
                           [EMPTY, PLAYER_ONE, EMPTY],
                           [EMPTY, EMPTY, PLAYER_ONE]])
        self.assertEqual(0, next_game_states[0].player_overwrites[PLAYER_ONE])

        def test_get_possible_moves_can_not_move_on_other_player(self):
            board = Board("""\
            2
            0
            0 0
            3 3
            2 0 0
            0 2 0
            0 0 1
            """)
            game = GameState(board)

            next_game_states = game.get_possible_moves_for_player(PLAYER_ONE, use_overwrite=True)
            self.assertEqual(0, len(next_game_states))

    def test_get_possible_moves_overwrite_on_own_player(self):
        board = Board("""\
        2
        1
        0 0
        3 3
        1 0 0
        0 2 0
        0 0 1
        """)
        game = GameState(board)

        next_game_states = game.get_possible_moves_for_player(PLAYER_ONE, use_overwrite=True)
        self.assertEqual(2, len(next_game_states))
        self.assert_board(next_game_states[0].board,
                          [[PLAYER_ONE, EMPTY, EMPTY],
                           [EMPTY, PLAYER_ONE, EMPTY],
                           [EMPTY, EMPTY, PLAYER_ONE]])
        self.assertEqual(0, next_game_states[0].player_overwrites[PLAYER_ONE])

    def test_get_possible_moves_expansion_only_with_overwrite(self):
        board = Board("""\
        2
        0
        0 0
        3 3
        0 0 0
        0 x 0
        0 0 0
        """)
        game = GameState(board)

        next_game_states = game.get_possible_moves_for_player(Field.PLAYER_ONE, use_overwrite=True)
        self.assertEqual(0, len(next_game_states))

    def test_get_possible_moves_holes_stop_move(self):
        board = Board("""\
        2
        0
        0 0
        3 3
        1 0 0
        0 2 -
        0 0 0
        """)
        game = GameState(board)

        next_game_states = game.get_possible_moves_for_player(PLAYER_ONE)
        self.assertEqual(1, len(next_game_states))
        self.assert_board(next_game_states[0].board,
                          [[PLAYER_ONE, EMPTY, EMPTY],
                           [EMPTY, PLAYER_ONE, HOLE],
                           [EMPTY, EMPTY, PLAYER_ONE]])

    def test_get_possible_moves_must_capture_minimum_one_enemy(self):
        board = Board("""\
        2
        1
        0 0
        3 3
        1 0 0
        0 0 0
        0 0 0
        """)
        game = GameState(board)

        next_game_states = game.get_possible_moves_for_player(PLAYER_ONE)
        self.assertEqual(0, len(next_game_states))

    def test_get_possible_moves_walk_through_transition(self):
        board = Board("""\
        2
        0
        0 0
        3 3
        0 2 0
        0 1 0
        0 0 0
        1 2 4 <-> 1 0 0
        """)
        game = GameState(board)

        next_game_states = game.get_possible_moves_for_player(PLAYER_ONE)
        self.assertEqual(1, len(next_game_states))
        self.assert_board(next_game_states[0].board,
                          [[EMPTY, PLAYER_ONE, EMPTY],
                           [EMPTY, PLAYER_ONE, EMPTY],
                           [EMPTY, PLAYER_ONE, EMPTY]])

    def test_get_possible_moves_walk_through_transition_hole(self):
        board = Board("""\
        2
        0
        0 0
        4 3
        0 2 0
        0 1 0
        0 0 0
        0 - 0
        1 2 4 <-> 1 0 0
        """)
        game = GameState(board)

        next_game_states = game.get_possible_moves_for_player(PLAYER_ONE)
        self.assertEqual(1, len(next_game_states))
        self.assert_board(next_game_states[0].board,
                          [[EMPTY, PLAYER_ONE, EMPTY],
                           [EMPTY, PLAYER_ONE, EMPTY],
                           [EMPTY, PLAYER_ONE, EMPTY],
                           [EMPTY, HOLE, EMPTY]])

    def test_get_possible_moves_walk_through_loop_transition(self):
        board = Board("""\
        2
        0
        0 0
        1 9
        1 - 2 - 0 - 2 - 1
        0 0 6 <-> 8 0 2
        2 0 6 <-> 0 0 2
        4 0 6 <-> 2 0 2
        6 0 6 <-> 4 0 2
        8 0 6 <-> 6 0 2
        """)
        game = GameState(board)

        next_game_states = game.get_possible_moves_for_player(PLAYER_ONE)
        self.assertEqual(1, len(next_game_states))
        self.assert_board(next_game_states[0].board,
                          [[PLAYER_ONE, HOLE, PLAYER_ONE, HOLE, PLAYER_ONE, HOLE, PLAYER_ONE, HOLE, PLAYER_ONE]])

    def test_get_possible_moves_bonus(self):
        board = Board("""\
        2
        1
        0 0
        3 3
        2 1 0
        - 2 -
        0 b 0
        """)
        game = GameState(board)

        next_game_states = game.get_possible_moves_for_player(PLAYER_ONE)
        self.assertEqual(2, len(next_game_states))

        self.assertEqual(next_game_states[0].player_bombs[PLAYER_ONE], 1)
        self.assertEqual(next_game_states[0].player_bombs[PLAYER_TWO], 0)
        self.assertEqual(next_game_states[0].player_overwrites[PLAYER_ONE], 1)
        self.assertEqual(next_game_states[0].player_overwrites[PLAYER_TWO], 1)

        self.assertEqual(next_game_states[1].player_bombs[PLAYER_ONE], 0)
        self.assertEqual(next_game_states[1].player_bombs[PLAYER_TWO], 0)
        self.assertEqual(next_game_states[1].player_overwrites[PLAYER_ONE], 2)
        self.assertEqual(next_game_states[1].player_overwrites[PLAYER_TWO], 1)

    def test_get_possible_moves_basic_choice(self):
        board = Board("""\
        2
        0
        0 0
        3 3
        2 1 0
        - 2 -
        0 c 0
        """)
        game = GameState(board)

        next_game_states = game.get_possible_moves_for_player(PLAYER_ONE)
        self.assertEqual(2, len(next_game_states))
        self.assert_board(next_game_states[0].board,
                          [[PLAYER_TWO, PLAYER_ONE, EMPTY],
                           [HOLE, PLAYER_ONE, HOLE],
                           [EMPTY, PLAYER_ONE, EMPTY]])
        self.assert_board(next_game_states[1].board,
                          [[PLAYER_ONE, PLAYER_TWO, EMPTY],
                           [HOLE, PLAYER_TWO, HOLE],
                           [EMPTY, PLAYER_TWO, EMPTY]])

    def test_get_possible_moves_three_player_choice(self):
        board = Board("""\
        3
        0
        0 0
        3 3
        2 1 3
        0 2 0
        0 c 0
        """)
        game = GameState(board)

        next_game_states = game.get_possible_moves_for_player(PLAYER_ONE)
        self.assertEqual(3, len(next_game_states))
        self.assert_board(next_game_states[0].board,
                          [[PLAYER_TWO, PLAYER_ONE, PLAYER_THREE],
                           [EMPTY, PLAYER_ONE, EMPTY],
                           [EMPTY, PLAYER_ONE, EMPTY]])
        self.assert_board(next_game_states[1].board,
                          [[PLAYER_ONE, PLAYER_TWO, PLAYER_THREE],
                           [EMPTY, PLAYER_TWO, EMPTY],
                           [EMPTY, PLAYER_TWO, EMPTY]])
        self.assert_board(next_game_states[2].board,
                          [[PLAYER_TWO, PLAYER_THREE, PLAYER_ONE],
                           [EMPTY, PLAYER_THREE, EMPTY],
                           [EMPTY, PLAYER_THREE, EMPTY]])

    def test_get_possible_moves_eight_player_choice(self):
        board = Board("""\
        8
        0
        0 0
        3 3
        2 1 3
        4 2 5
        6 c 7
        """)
        game = GameState(board)

        next_game_states = game.get_possible_moves_for_player(PLAYER_ONE)
        self.assertEqual(8, len(next_game_states))
        self.assert_board(next_game_states[0].board,
                          [[PLAYER_TWO, PLAYER_ONE, PLAYER_THREE],
                           [PLAYER_FOUR, PLAYER_ONE, PLAYER_FIVE],
                           [PLAYER_SIX, PLAYER_ONE, PLAYER_SEVEN]])
        self.assert_board(next_game_states[1].board,
                          [[PLAYER_ONE, PLAYER_TWO, PLAYER_THREE],
                           [PLAYER_FOUR, PLAYER_TWO, PLAYER_FIVE],
                           [PLAYER_SIX, PLAYER_TWO, PLAYER_SEVEN]])
        self.assert_board(next_game_states[2].board,
                          [[PLAYER_TWO, PLAYER_THREE, PLAYER_ONE],
                           [PLAYER_FOUR, PLAYER_THREE, PLAYER_FIVE],
                           [PLAYER_SIX, PLAYER_THREE, PLAYER_SEVEN]])
        self.assert_board(next_game_states[3].board,
                          [[PLAYER_TWO, PLAYER_FOUR, PLAYER_THREE],
                           [PLAYER_ONE, PLAYER_FOUR, PLAYER_FIVE],
                           [PLAYER_SIX, PLAYER_FOUR, PLAYER_SEVEN]])
        self.assert_board(next_game_states[4].board,
                          [[PLAYER_TWO, PLAYER_FIVE, PLAYER_THREE],
                           [PLAYER_FOUR, PLAYER_FIVE, PLAYER_ONE],
                           [PLAYER_SIX, PLAYER_FIVE, PLAYER_SEVEN]])
        self.assert_board(next_game_states[5].board,
                          [[PLAYER_TWO, PLAYER_SIX, PLAYER_THREE],
                           [PLAYER_FOUR, PLAYER_SIX, PLAYER_FIVE],
                           [PLAYER_ONE, PLAYER_SIX, PLAYER_SEVEN]])
        self.assert_board(next_game_states[6].board,
                          [[PLAYER_TWO, PLAYER_SEVEN, PLAYER_THREE],
                           [PLAYER_FOUR, PLAYER_SEVEN, PLAYER_FIVE],
                           [PLAYER_SIX, PLAYER_SEVEN, PLAYER_ONE]])
        self.assert_board(next_game_states[7].board,
                          [[PLAYER_TWO, PLAYER_EIGHT, PLAYER_THREE],
                           [PLAYER_FOUR, PLAYER_EIGHT, PLAYER_FIVE],
                           [PLAYER_SIX, PLAYER_EIGHT, PLAYER_SEVEN]])

    def test_get_possible_moves_basic_inversion(self):
        board = Board("""\
        2
        0
        0 0
        3 3
        0 0 1
        0 2 0
        i 0 0
        """)
        game = GameState(board)

        next_game_states = game.get_possible_moves_for_player(PLAYER_ONE)
        self.assertEqual(1, len(next_game_states))
        self.assert_board(next_game_states[0].board,
                          [[EMPTY, EMPTY, PLAYER_TWO],
                           [EMPTY, PLAYER_TWO, EMPTY],
                           [PLAYER_TWO, EMPTY, EMPTY]])

    def test_get_possible_moves_inversion_with_transition(self):
        board = Board("""\
        2
        0
        0 0
        3 3
        2 - 1
        2 2 1
        i - -
        0 0 0 <-> 0 1 6
        """)
        game = GameState(board)

        next_game_states = game.get_possible_moves_for_player(PLAYER_ONE)
        self.assertEqual(1, len(next_game_states))
        self.assert_board(next_game_states[0].board,
                          [[PLAYER_TWO, HOLE, PLAYER_TWO],
                           [PLAYER_TWO, PLAYER_TWO, PLAYER_TWO],
                           [PLAYER_TWO, HOLE, HOLE]])

    def test_get_possible_moves_three_player_inversion(self):
        board = Board("""\
        3
        0
        0 0
        3 3
        0 0 1
        0 2 3
        i 0 0
        """)
        game = GameState(board)

        next_game_states = game.get_possible_moves_on_position((0, 2), player=PLAYER_ONE)
        self.assertEqual(1, len(next_game_states))
        self.assert_board(next_game_states[0].board,
                          [[EMPTY, EMPTY, PLAYER_TWO],
                           [EMPTY, PLAYER_TWO, PLAYER_ONE],
                           [PLAYER_TWO, EMPTY, EMPTY]])

    def test_get_possible_moves_four_player_inversion(self):
        board = Board("""\
        4
        0
        0 0
        3 3
        0 4 1
        0 2 3
        i 0 0
        """)
        game = GameState(board)

        next_game_states = game.get_possible_moves_on_position((0, 2), player=PLAYER_ONE)
        self.assertEqual(1, len(next_game_states))
        self.assert_board(next_game_states[0].board,
                          [[EMPTY, PLAYER_ONE, PLAYER_TWO],
                           [EMPTY, PLAYER_TWO, PLAYER_FOUR],
                           [PLAYER_TWO, EMPTY, EMPTY]])

    def test_get_possible_moves_five_player_inversion(self):
        board = Board("""\
        5
        0
        0 0
        3 3
        5 4 1
        0 2 3
        i 0 0
        """)
        game = GameState(board)

        next_game_states = game.get_possible_moves_on_position((0, 2), player=PLAYER_ONE)
        self.assertEqual(1, len(next_game_states))
        self.assert_board(next_game_states[0].board,
                          [[PLAYER_ONE, PLAYER_FIVE, PLAYER_TWO],
                           [EMPTY, PLAYER_TWO, PLAYER_FOUR],
                           [PLAYER_TWO, EMPTY, EMPTY]])

    def test_get_possible_moves_six_player_inversion(self):
        board = Board("""\
        6
        0
        0 0
        3 3
        5 4 1
        6 2 3
        i 0 0
        """)
        game = GameState(board)

        next_game_states = game.get_possible_moves_on_position((0, 2), player=PLAYER_ONE)
        self.assertEqual(1, len(next_game_states))
        self.assert_board(next_game_states[0].board,
                          [[PLAYER_SIX, PLAYER_FIVE, PLAYER_TWO],
                           [PLAYER_ONE, PLAYER_TWO, PLAYER_FOUR],
                           [PLAYER_TWO, EMPTY, EMPTY]])

    def test_get_possible_moves_seven_player_inversion(self):
        board = Board("""\
        7
        0
        0 0
        3 3
        5 4 1
        6 2 3
        i 7 0
        """)
        game = GameState(board)

        next_game_states = game.get_possible_moves_on_position((0, 2), player=PLAYER_ONE)
        self.assertEqual(1, len(next_game_states))
        self.assert_board(next_game_states[0].board,
                          [[PLAYER_SIX, PLAYER_FIVE, PLAYER_TWO],
                           [PLAYER_SEVEN, PLAYER_TWO, PLAYER_FOUR],
                           [PLAYER_TWO, PLAYER_ONE, EMPTY]])

    def test_get_possible_moves_eight_player_inversion(self):
        board = Board("""\
        8
        0
        0 0
        3 3
        5 4 1
        6 2 3
        i 7 8
        """)
        game = GameState(board)

        next_game_states = game.get_possible_moves_on_position((0, 2), player=PLAYER_ONE)
        self.assertEqual(1, len(next_game_states))
        self.assert_board(next_game_states[0].board,
                          [[PLAYER_SIX, PLAYER_FIVE, PLAYER_TWO],
                           [PLAYER_SEVEN, PLAYER_TWO, PLAYER_FOUR],
                           [PLAYER_TWO, PLAYER_EIGHT, PLAYER_ONE]])

    def test_get_possible_moves_bomb_zero(self):
        board = Board("""\
        2
        0
        2 0
        3 3
        0 1 0
        0 2 0
        0 0 0
        """)
        game = GameState(board)
        game.bomb_phase = True

        next_game_states = game.get_possible_bomb_move_on_position((1, 1), player=PLAYER_ONE)
        self.assertEqual(1, len(next_game_states))
        self.assert_board(next_game_states[0].board,
                          [[EMPTY, PLAYER_ONE, EMPTY],
                           [EMPTY, HOLE, EMPTY],
                           [EMPTY, EMPTY, EMPTY]])
        self.assertEqual(next_game_states[0].player_bombs[PLAYER_ONE], 1)
        self.assertEqual(next_game_states[0].player_bombs[PLAYER_TWO], 2)

    def test_get_possible_moves_bomb_one(self):
        board = Board("""\
        2
        0
        2 1
        3 3
        0 0 0
        - - 0
        0 0 0
        """)
        game = GameState(board)
        game.bomb_phase = True

        next_game_states = game.get_possible_bomb_move_on_position((0, 0), player=PLAYER_ONE)
        self.assertEqual(1, len(next_game_states))
        self.assert_board(next_game_states[0].board,
                          [[HOLE, HOLE, EMPTY],
                           [HOLE, HOLE, EMPTY],
                           [EMPTY, EMPTY, EMPTY]])
        self.assertEqual(next_game_states[0].player_bombs[PLAYER_ONE], 1)
        self.assertEqual(next_game_states[0].player_bombs[PLAYER_TWO], 2)

    def test_get_possible_moves_bomb_two(self):
        board = Board("""\
        2
        0
        2 2
        3 3
        0 0 0
        - - 0
        0 0 0
        """)
        game = GameState(board)
        game.bomb_phase = True

        next_game_states = game.get_possible_bomb_move_on_position((0, 0), player=PLAYER_ONE)
        self.assertEqual(1, len(next_game_states))
        self.assert_board(next_game_states[0].board,
                          [[HOLE, HOLE, HOLE],
                           [HOLE, HOLE, HOLE],
                           [EMPTY, EMPTY, EMPTY]])
        self.assertEqual(next_game_states[0].player_bombs[PLAYER_ONE], 1)
        self.assertEqual(next_game_states[0].player_bombs[PLAYER_TWO], 2)

    def test_get_possible_moves_bomb_transition(self):
        board = Board("""\
        2
        0
        2 1
        3 3
        0 0 0
        - - 0
        0 0 0
        0 0 6 <-> 2 2 2
        """)
        game = GameState(board)
        game.bomb_phase = True

        next_game_states = game.get_possible_bomb_move_on_position((0, 0), player=PLAYER_ONE)
        self.assertEqual(1, len(next_game_states))
        self.assert_board(next_game_states[0].board,
                          [[HOLE, HOLE, EMPTY],
                           [HOLE, HOLE, EMPTY],
                           [EMPTY, EMPTY, HOLE]])
        self.assertEqual(next_game_states[0].player_bombs[PLAYER_ONE], 1)
        self.assertEqual(next_game_states[0].player_bombs[PLAYER_TWO], 2)

    def test_get_possible_moves_boese01(self):
        board = Board("""\
        2
        0
        0 0
        8 11
        - - - - - - 0 - - - - 
        - - - - - - 2 - - - - 
        - 2 2 2 2 2 2 2 1 0 0 
        - - - - - - 2 - - - - 
        - - - - - - 2 - - - - 
        - - - - - - 2 - - - - 
        - - - - - - 2 - - - - 
        - - - - - - 2 - - - - 
        1 2 6 <-> 6 7 4
        """)
        game = GameState(board)

        next_game_states = game.get_possible_moves_for_player(PLAYER_ONE)
        self.assertEqual(1, len(next_game_states))
        self.assert_board(next_game_states[0].board,
                          [[HOLE, HOLE, HOLE, HOLE, HOLE, HOLE, PLAYER_ONE, HOLE, HOLE, HOLE, HOLE],
                           [HOLE, HOLE, HOLE, HOLE, HOLE, HOLE, PLAYER_ONE, HOLE, HOLE, HOLE, HOLE],
                           [HOLE, PLAYER_ONE, PLAYER_ONE, PLAYER_ONE, PLAYER_ONE, PLAYER_ONE, PLAYER_ONE, PLAYER_ONE, PLAYER_ONE, EMPTY, EMPTY],
                           [HOLE, HOLE, HOLE, HOLE, HOLE, HOLE, PLAYER_ONE, HOLE, HOLE, HOLE, HOLE],
                           [HOLE, HOLE, HOLE, HOLE, HOLE, HOLE, PLAYER_ONE, HOLE, HOLE, HOLE, HOLE],
                           [HOLE, HOLE, HOLE, HOLE, HOLE, HOLE, PLAYER_ONE, HOLE, HOLE, HOLE, HOLE],
                           [HOLE, HOLE, HOLE, HOLE, HOLE, HOLE, PLAYER_ONE, HOLE, HOLE, HOLE, HOLE],
                           [HOLE, HOLE, HOLE, HOLE, HOLE, HOLE, PLAYER_ONE, HOLE, HOLE, HOLE, HOLE]])

    def test_get_possible_moves_boese02(self):
        board = Board("""\
        2
        0
        0 0
        2 10
        2 2 2 0 2 2 2 2 2 2 
        - - - - - 1 2 0 - - 
        0 0 6 <-> 9 0 2
        """)
        game = GameState(board)

        next_game_states = game.get_possible_moves_on_position((3, 0), player=PLAYER_ONE)
        self.assertEqual(0, len(next_game_states))

    def test_get_possible_moves_boese03(self):
        board = Board("""\
        2
        0
        0 0
        2 10
        2 2 2 1 0 2 2 2 2 2 
        - - - - 0 1 2 0 - -
        0 0 6 <-> 9 0 2
        """)
        game = GameState(board)

        next_game_states = game.get_possible_moves_on_position((4, 0), player=PLAYER_ONE)
        self.assertEqual(1, len(next_game_states))
        self.assert_board(next_game_states[0].board,
                          [[PLAYER_ONE, PLAYER_ONE, PLAYER_ONE, PLAYER_ONE, PLAYER_ONE, PLAYER_ONE, PLAYER_ONE, PLAYER_ONE, PLAYER_ONE, PLAYER_ONE],
                           [HOLE, HOLE, HOLE, HOLE, EMPTY, PLAYER_ONE, PLAYER_TWO, EMPTY, HOLE, HOLE]])

    def test_get_possible_moves_boese04(self):
        board = Board("""\
        2
        0
        0 0
        4 3
        0 0 0 
        0 2 2 
        1 0 2 
        2 2 2 
        2 3 4 <-> 2 3 2
        0 2 5 <-> 0 3 6
        """)
        game = GameState(board)

        next_game_states = game.get_possible_moves_on_position((2, 0), player=PLAYER_ONE)
        self.assertEqual(1, len(next_game_states))
        self.assert_board(next_game_states[0].board,
                          [[EMPTY, EMPTY, PLAYER_ONE],
                           [EMPTY, PLAYER_ONE, PLAYER_ONE],
                           [PLAYER_ONE, EMPTY, PLAYER_ONE],
                           [PLAYER_ONE, PLAYER_ONE, PLAYER_ONE]])

    def test_get_possible_moves_boese05(self):
        board = Board("""\
        2
        0
        0 0
        4 3
        2 2 1 
        0 2 2 
        0 0 2 
        2 2 2 
        2 3 4 <-> 2 3 2
        0 2 5 <-> 0 3 6
        0 0 6 <-> 0 2 6
        """)
        game = GameState(board)

        next_game_states = game.get_possible_moves_on_position((0, 2), player=PLAYER_ONE)
        self.assertEqual(1, len(next_game_states))
        self.assert_board(next_game_states[0].board,
                          [[PLAYER_ONE, PLAYER_ONE, PLAYER_ONE],
                           [EMPTY, PLAYER_ONE, PLAYER_ONE],
                           [PLAYER_ONE, EMPTY, PLAYER_ONE],
                           [PLAYER_ONE, PLAYER_ONE, PLAYER_ONE]])

    def test_get_possible_moves_boese06(self):
        board = Board("""\
        2
        0
        0 0
        4 3
        2 2 1 
        0 2 2 
        0 0 2 
        2 2 1 
        2 3 4 <-> 2 3 2
        0 2 5 <-> 0 3 6
        0 0 6 <-> 0 2 6
        """)
        game = GameState(board)

        next_game_states = game.get_possible_moves_on_position((0, 2), player=PLAYER_ONE)
        self.assertEqual(1, len(next_game_states))
        self.assert_board(next_game_states[0].board,
                          [[PLAYER_ONE, PLAYER_ONE, PLAYER_ONE],
                           [EMPTY, PLAYER_ONE, PLAYER_TWO],
                           [PLAYER_ONE, EMPTY, PLAYER_TWO],
                           [PLAYER_ONE, PLAYER_ONE, PLAYER_ONE]])

    def test_get_possible_moves_boese07(self):
        board = Board("""\
        2
        1
        0 0
        8 11
        - - - - - - 1 - - - - 
        - - - - - - 2 - - - - 
        - 2 2 2 2 2 2 2 1 1 0 
        - - - - - - 2 - - - - 
        - - - - - - 2 - - - - 
        - - - - - - 2 - - - - 
        - - - - - - 2 - - - - 
        - - - - - - 2 - - - - 
        1 2 6 <-> 6 7 4
        """)
        game = GameState(board)

        next_game_states = game.get_possible_moves_on_position((6, 2), player=PLAYER_ONE, use_overwrite=True)
        self.assertEqual(1, len(next_game_states))
        self.assert_board(next_game_states[0].board,
                          [[HOLE, HOLE, HOLE, HOLE, HOLE, HOLE, PLAYER_ONE, HOLE, HOLE, HOLE, HOLE],
                           [HOLE, HOLE, HOLE, HOLE, HOLE, HOLE, PLAYER_ONE, HOLE, HOLE, HOLE, HOLE],
                           [HOLE, PLAYER_TWO, PLAYER_TWO, PLAYER_TWO, PLAYER_TWO, PLAYER_TWO, PLAYER_ONE, PLAYER_ONE, PLAYER_ONE, PLAYER_ONE, EMPTY],
                           [HOLE, HOLE, HOLE, HOLE, HOLE, HOLE, PLAYER_TWO, HOLE, HOLE, HOLE, HOLE],
                           [HOLE, HOLE, HOLE, HOLE, HOLE, HOLE, PLAYER_TWO, HOLE, HOLE, HOLE, HOLE],
                           [HOLE, HOLE, HOLE, HOLE, HOLE, HOLE, PLAYER_TWO, HOLE, HOLE, HOLE, HOLE],
                           [HOLE, HOLE, HOLE, HOLE, HOLE, HOLE, PLAYER_TWO, HOLE, HOLE, HOLE, HOLE],
                           [HOLE, HOLE, HOLE, HOLE, HOLE, HOLE, PLAYER_TWO, HOLE, HOLE, HOLE, HOLE]])

    def test_get_possible_moves_boese_lost_trace(self):
        board = Board("""\
        2
        0
        0 0
        8 9
        - - - - - - - - -
        - - 2 - - - 2 - -
        - 2 2 2 0 2 2 2 -
        - - 2 - - - 2 - -
        - - 2 - - - 2 - -
        - - 2 - - - 2 - -
        - - 1 - - - 1 - -
        - - - - - - - - -
        7 2 2 <-> 2 1 0
        1 2 6 <-> 6 1 0
        """)
        game = GameState(board)

        next_game_states = game.get_possible_moves_on_position((4, 2), player=PLAYER_ONE, use_overwrite=True)
        self.assertEqual(1, len(next_game_states))
        self.assert_board(next_game_states[0].board,
                          [[HOLE, HOLE, HOLE, HOLE, HOLE, HOLE, HOLE, HOLE, HOLE],
                           [HOLE, HOLE, PLAYER_ONE, HOLE, HOLE, HOLE, PLAYER_ONE, HOLE, HOLE],
                           [HOLE, PLAYER_ONE, PLAYER_ONE, PLAYER_ONE, PLAYER_ONE, PLAYER_ONE, PLAYER_ONE, PLAYER_ONE, HOLE],
                           [HOLE, HOLE, PLAYER_ONE, HOLE, HOLE, HOLE, PLAYER_ONE, HOLE, HOLE],
                           [HOLE, HOLE, PLAYER_ONE, HOLE, HOLE, HOLE, PLAYER_ONE, HOLE, HOLE],
                           [HOLE, HOLE, PLAYER_ONE, HOLE, HOLE, HOLE, PLAYER_ONE, HOLE, HOLE],
                           [HOLE, HOLE, PLAYER_ONE, HOLE, HOLE, HOLE, PLAYER_ONE, HOLE, HOLE],
                           [HOLE, HOLE, HOLE, HOLE, HOLE, HOLE, HOLE, HOLE, HOLE]])

    def test_get_possible_moves_boese_reflect(self):
        board = Board("""\
        2
        1
        0 0
        9 4
        - - - - 
        - - - 2 
        - - - 2 
        - - - 2 
        - - - 1 
        - - - 0 
        - - - 0 
        - - - 0 
        - - - 0 
        3 1 0 <-> 3 1 0
        """)
        game = GameState(board)

        next_game_states = game.get_possible_moves_on_position((3, 4), player=PLAYER_ONE, use_overwrite=True)
        self.assertEqual(0, len(next_game_states))

    def assert_board(self, board, target_board):
        for y in range(board.height):
            for x in range(board.width):
                if board[(x, y)] != target_board[y][x]:
                    print((x, y))
                self.assertEqual(board[(x, y)], target_board[y][x])


if __name__ == '__main__':
    unittest.main()
