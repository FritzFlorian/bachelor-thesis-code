# cython: profile=True
import numpy as np
from reversi.game_core import Field


def input(evaluation, calculate_target=False):
    N_RAW_VALUES = 3
    normal_evaluation = evaluation.convert_to_normal()

    game_state = normal_evaluation.game_state
    possible_moves = normal_evaluation.possible_moves
    board = game_state.board

    board_array = board.get_raw_board()
    input_array = np.eye(N_RAW_VALUES + 1)[board_array]

    # Mark all possible moves in the last one hot layer
    if not possible_moves:
        next_game_states = game_state.get_next_possible_moves()
        possible_moves = [next_game_state.last_move[1] for next_game_state in next_game_states]

    for possible_move in possible_moves:
        x, y = possible_move[1]
        input_array[y, x, N_RAW_VALUES] = 1

    if not calculate_target:
        return input_array, None

    value_outputs = np.array([normal_evaluation.expected_result[Field.PLAYER_ONE]])
    prob_outputs = np.zeros([8 * 8])

    for move, prob in normal_evaluation.probabilities.items():
        x, y = move[1]
        prob_outputs[y * 8 + x] = prob

    target_array = np.concatenate((prob_outputs, value_outputs), axis=0)

    return input_array, target_array


def output(evaluation, output_array):
    for move, prob in evaluation.probabilities.items():
        x, y = move[1]
        evaluation.probabilities[move] = output_array[y * 8 + x]

    evaluation.expected_result[Field.PLAYER_ONE] = output_array[8 * 8]
    evaluation.expected_result[Field.PLAYER_TWO] = -output_array[8 * 8]

    return evaluation
