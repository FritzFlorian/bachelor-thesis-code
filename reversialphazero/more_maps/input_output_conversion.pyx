# cython: profile=True
import numpy as np
from reversi.game_core import Field

# Number of values for one hot encoding (field, player_one, player_two)
N_RAW_VALUES = 4
# The size of the neural network internal board representation
# We wil embed our smaller board into it.
NN_BOARD_SIZE = 12


def input(evaluation, calculate_target=False):
    cdef int nn_board_size, n_raw_values
    cdef int embed_x, embed_y
    cdef int y, x, field
    cdef int empty_field, hole_field
    cdef int BOARD_SIZE, MAX_TRANSLATION

    # Calculate on how much we can maximally translate the board.
    # We only accept square maps here!!!
    BOARD_SIZE = evaluation.game_state.original_game_state.board.width
    MAX_TRANSLATION = NN_BOARD_SIZE - BOARD_SIZE

    nn_board_size = NN_BOARD_SIZE
    n_raw_values = N_RAW_VALUES
    empty_field = Field.EMPTY
    hole_field = Field.HOLE

    embed_x = np.random.randint(0, MAX_TRANSLATION + 1)
    embed_y = np.random.randint(0, MAX_TRANSLATION + 1)
    evaluation.embedding_position = (embed_x, embed_y)
    normal_evaluation = evaluation.convert_to_normal()

    game_state = normal_evaluation.game_state.original_game_state
    board = game_state.board

    # Gen an board with only holes
    nn_sized_board_array = np.zeros([nn_board_size, nn_board_size], dtype=np.int8)
    # Get the board's array to embed it
    board_array = board.get_raw_board()
    # Now we embed the board array into the bigger nn array
    nn_sized_board_array[embed_x:embed_x + BOARD_SIZE, embed_y:embed_y + BOARD_SIZE] = board_array
    # Last step is to one hot encode everything
    input_array = np.zeros([nn_board_size, nn_board_size, n_raw_values], dtype=np.int8)
    for y in range(nn_board_size):
        for x in range(nn_board_size):
            field = nn_sized_board_array[y][x]
            if field != hole_field:
                input_array[y, x, 0] = 1
                if field != empty_field:
                    input_array[y, x, field] = 1

    # Mark all possible moves in the last one hot layer
    for move, _ in normal_evaluation.probabilities.items():
        x, y = move.iternal_tuple[1]
        input_array[y + embed_y, x + embed_x, -1] = 1

    if not calculate_target:
        return input_array, None
    value_outputs = np.array([normal_evaluation.expected_result[Field.PLAYER_ONE]])

    prob_outputs = np.zeros([nn_board_size * nn_board_size])

    for move, prob in normal_evaluation.probabilities.items():
        x, y = move.internal_tuple[1]
        prob_outputs[(y + embed_y) * nn_board_size + (x + embed_x)] = prob

    target_array = np.concatenate((prob_outputs, value_outputs), axis=0)

    return input_array, target_array


def output(evaluation, output_array):
    embed_x, embed_y = evaluation.embedding_position

    for move, prob in evaluation.probabilities.items():
        x, y = move.internal_tuple[1]
        evaluation.probabilities[move] = output_array[(y + embed_y) * NN_BOARD_SIZE + (x + embed_x)]

    evaluation.expected_result[Field.PLAYER_ONE] = output_array[NN_BOARD_SIZE * NN_BOARD_SIZE]
    evaluation.expected_result[Field.PLAYER_TWO] = -output_array[NN_BOARD_SIZE * NN_BOARD_SIZE]

    return evaluation
