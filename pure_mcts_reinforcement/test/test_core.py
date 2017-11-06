import unittest
from reversi.game_core import Board, GameState, Field
from pure_mcts_reinforcement.core import Evaluation


class TestEvaluation(unittest.TestCase):
    def test_normal_form_conversion(self):
        board = Board("""\
        3
        1
        0 0
        3 3
        0 1 2
        0 3 0
        3 3 0
        """)

        game_state = GameState(board)
        game_state = game_state.get_next_possible_moves()[0]

        evaluation = Evaluation(game_state)
        evaluation.probabilities[(Field.PLAYER_ONE, (1, 2), None)] = 1.0
        evaluation.expected_result[Field.PLAYER_ONE] = 0.7
        evaluation.expected_result[Field.PLAYER_TWO] = 0.2
        evaluation.expected_result[Field.PLAYER_THREE] = 0.1

        evaluation.convert_to_normal()
        self.assertEqual(evaluation.game_state.next_player(), Field.PLAYER_ONE)
        self.assertEqual(evaluation.expected_result[Field.PLAYER_THREE], 0.7)
        self.assertFalse((Field.PLAYER_ONE, (1, 2), None) in evaluation.probabilities)
        self.assertEqual(evaluation.probabilities[(Field.PLAYER_THREE, (1, 2), None)], 1.0)
        self.assertEqual(evaluation.game_state.board[(1, 0)], Field.PLAYER_THREE)
        self.assertEqual(evaluation.game_state.player_overwrites[Field.PLAYER_THREE], 0)
        self.assertEqual(evaluation.game_state.player_overwrites[Field.PLAYER_ONE], 1)

        evaluation.convert_from_normal()
        self.assertEqual(evaluation.game_state.next_player(), Field.PLAYER_TWO)
        self.assertEqual(evaluation.expected_result[Field.PLAYER_ONE], 0.7)
        self.assertFalse((Field.PLAYER_THREE, (1, 2), None) in evaluation.probabilities)
        self.assertEqual(evaluation.probabilities[(Field.PLAYER_ONE, (1, 2), None)], 1.0)
        self.assertEqual(evaluation.game_state.board[(1, 0)], Field.PLAYER_ONE)
        self.assertEqual(evaluation.game_state.player_overwrites[Field.PLAYER_ONE], 0)
        self.assertEqual(evaluation.game_state.player_overwrites[Field.PLAYER_THREE], 1)


if __name__ == '__main__':
    unittest.main()
