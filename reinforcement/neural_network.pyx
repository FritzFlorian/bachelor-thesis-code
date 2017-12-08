# cython: profile=True
"""Functionality directly related to the Neural Networks used.

This includes the abstract base class for every custom Neural Network and conversion functions
used for the input/output of various neural networks."""
import numpy as np
from reversi.game_core import Field


class NeuralNetwork:
    """Wrapper that represents a single neural network instance.

    This is intended to abstract away the actual creation, training and execution of the neural network.
    This should hopefully also allow to re-use major parts of the code for different network structures.

    The network is not responsible for managing its scope/tensorflow graph, this should be done
    by the code that uses and executes it."""
    def construct_network(self):
        raise NotImplementedError("Add the construction of your custom graph structure.")

    def init_network(self):
        raise NotImplementedError("Run initialisation code for your network.")

    def input_conversion_function(self):
        raise NotImplementedError("Return a reference to the function converting evaluations to input for thin NN.")

    def output_conversion_function(self):
        raise NotImplementedError("Return a reference to the function filling outputs into evaluations.")

    def execute_batch(self, sess, input_arrays):
        raise NotImplementedError("Add implementation that takes prepared input arrays and executes them as a batch.")

    def train_batch(self, sess, input_arrays, output_arrays):
        raise NotImplementedError("Add implementation that executes one batch training step.")

    def save_weights(self, sess, filename):
        raise NotImplementedError("Add implementation that saves the weights of this network to a checkpoint.")

    def load_weights(self, sess, filename):
        raise NotImplementedError("Add implementation that loads the weights of this network to a checkpoint.")

    def log_loss(self, sess, tf_file_writer, input_arrays, target_arrays, epoch):
        raise NotImplementedError("Add implementation to write average losses to the stats file and return them.")

# Input/Output conversion to/from evaluations to/from raw neural network input/output.
# These methods need to live somewhere separate, as we not need to import an tensorflow
# code in our main code, so we can execute it using pypy.


def simple_8_by_8_input(evaluation, calculate_target=False):
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


def simple_8_by_8_output(evaluation, output_array):
    for move, prob in evaluation.probabilities.items():
        x, y = move[1]
        evaluation.probabilities[move] = output_array[y * 8 + x]

    evaluation.expected_result[Field.PLAYER_ONE] = output_array[8 * 8]
    evaluation.expected_result[Field.PLAYER_TWO] = -output_array[8 * 8]

    return evaluation


