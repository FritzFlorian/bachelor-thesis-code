# cython: profile=True
"""Functionality directly related to the Neural Networks used.

This includes the abstract base class for every custom Neural Network and conversion functions
used for the input/output of various neural networks."""


class NeuralNetwork:
    """Wrapper that represents a single neural network instance.

    This is intended to abstract away the actual creation, training and execution of the neural network.
    This should hopefully also allow to re-use major parts of the code for different network structures.

    The network is not responsible for managing its scope/tensorflow graph, this should be done
    by the code that uses and executes it."""
    def construct_network(self, sess, graph):
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

    def log_training_progress(self, sess, tf_file_writer, input_arrays, target_arrays, training_batch):
        raise NotImplementedError("Add implementation to write stats on the current training progress.")

