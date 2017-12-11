# cython: profile=True
import numpy as np
from reversi.game_core import Field

# Number of values for one hot encoding (field, player_one, player_two)
N_RAW_VALUES = 4
# The size of the neural network internal board representation
# We wil embed our smaller board into it.
NN_BOARD_SIZE = 12
# The normal board size.
BOARD_SIZE = 8
# Now we now how much we can translate the board.
BORDER = 1
MAX_TRANSLATION = NN_BOARD_SIZE - BOARD_SIZE - 2 * BORDER

def input(evaluation, calculate_target=False):
    embed_x = np.random.randint(0, MAX_TRANSLATION + 1) + BORDER
    embed_y = np.random.randint(0, MAX_TRANSLATION + 1) + BORDER
    evaluation.embedding_position = (embed_x, embed_y)
    normal_evaluation = evaluation.convert_to_normal()

    game_state = normal_evaluation.game_state
    possible_moves = normal_evaluation.possible_moves
    board = game_state.board

    # Gen an board with only holes
    nn_sized_board_array = np.zeros([NN_BOARD_SIZE, NN_BOARD_SIZE], dtype=np.int8)
    # Get the board's array to embed it
    board_array = board.get_raw_board()
    # Now we embed the board array into the bigger nn array
    nn_sized_board_array[embed_x:embed_x + BOARD_SIZE, embed_y:embed_y + BOARD_SIZE] = board_array
    # Last step is to one hot encode everything
    input_array = np.eye(N_RAW_VALUES)[nn_sized_board_array]

    # Mark all possible moves in the last one hot layer
    if not possible_moves:
        next_game_states = game_state.get_next_possible_moves()
        possible_moves = [next_game_state.last_move[1] for next_game_state in next_game_states]

    for possible_move in possible_moves:
        x, y = possible_move[1]
        input_array[y + embed_y, x + embed_x, -1] = 1

    if not calculate_target:
        return input_array, None

    value_outputs = np.array([normal_evaluation.expected_result[Field.PLAYER_ONE]])
    prob_outputs = np.zeros([NN_BOARD_SIZE * NN_BOARD_SIZE])

    for move, prob in normal_evaluation.probabilities.items():
        x, y = move[1]
        prob_outputs[(y + embed_y) * NN_BOARD_SIZE + (x + embed_x)] = prob

    target_array = np.concatenate((prob_outputs, value_outputs), axis=0)

    return input_array, target_array


def output(evaluation, output_array):
    embed_x, embed_y = evaluation.embedding_position

    for move, prob in evaluation.probabilities.items():
        x, y = move[1]
        evaluation.probabilities[move] = output_array[(y + embed_y) * NN_BOARD_SIZE + (x + embed_x)]

    evaluation.expected_result[Field.PLAYER_ONE] = output_array[NN_BOARD_SIZE * NN_BOARD_SIZE]
    evaluation.expected_result[Field.PLAYER_TWO] = -output_array[NN_BOARD_SIZE * NN_BOARD_SIZE]

    return evaluation
