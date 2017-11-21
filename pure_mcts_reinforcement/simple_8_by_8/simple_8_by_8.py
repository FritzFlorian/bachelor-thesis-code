import pure_mcts_reinforcement.core as core
import pure_mcts_reinforcement.distribution as distribution
import tensorflow as tf
import numpy as np
import sklearn.preprocessing
from reversi.game_core import Field, Board, GameState
import multiprocessing
import os
import time
import random


BOARD_HEIGHT = 8
BOARD_WIDTH = 8

# Number of different possible states/contents of a
# single field on the board.
N_RAW_VALUES = 3
FLOAT = tf.float32

L2_LOSS_WEIGHT = 0.005

# Number of games played to gather training data per epoch (per NN configuration)
GAMES_PER_EPOCH = 16
SIMULATIONS_PER_GAME_TURN = 80

TRAINING_BATCHES_PER_EPOCH = 1_500
BATCH_SIZE = 64

N_EPOCHS = 10

CHECKPOINT_FOLDER = 'checkpoints'
DATA_FOLDER = 'data'
BEST_CHECKPOINT_FOLDER = 'best-checkpoint'

N_EVALUATION_GAMES = 16
NEEDED_AVG_SCORE = 0.05

N_AI_EVALUATION_GAMES = 16



def main():
    # FIXME: Tensorflow issue when using multiprocesning
    multiprocessing.set_start_method('spawn', True)

    # Load our map, for now we play only on one
    with open('simple_8_by_8.map') as file:
        board = Board(file.read())
    initial_game_state = GameState(board)

    best_model_dir = BEST_CHECKPOINT_FOLDER
    best_model_file = os.path.join(best_model_dir, 'checkpoint.ckpt')

    # Make sure we got a point to start training
    if not os.path.exists(best_model_dir):
        create_directory(best_model_dir)
        neural_network = SimpleNeuralNetwork()
        with tf.Graph().as_default():
            neural_network.construct_network()
            with tf.Session() as sess:
                neural_network.init_network()
                neural_network.save_weights(sess, best_model_file)

    # Current run...
    run_dir = './run_{}'.format(round(time.time() * 1000))

    # Keep Track of the AI evaluations.
    # We hope to see some sort of improvement over time here.
    create_directory(os.path.join(run_dir, CHECKPOINT_FOLDER))
    stats_file = os.path.join(run_dir, 'stats.csv')
    with open(stats_file, 'w') as file:
        file.write('epoch,nn_score,ai_score,nn_stones,ai_stones\n')

    # Train for some epochs.
    # Each epoch equals one round of selfplay and training.
    for epoch in range(N_EPOCHS):
        current_data_dir = os.path.join(run_dir, DATA_FOLDER, 'epoch-{0:05d}'.format(epoch))
        current_ckpt_dir = os.path.join(run_dir, CHECKPOINT_FOLDER, 'ckpt-{0:05d}'.format(epoch))
        current_ckpt_file = os.path.join(current_ckpt_dir, 'checkpoint.ckpt')
        create_directory(current_ckpt_dir)

        # Make sure to add training data to our training executor.
        reuse_old_training_data = os.path.exists(current_data_dir)
        training_executor = core.TrainingExecutor(SimpleNeuralNetwork(), best_model_file, current_data_dir)
        training_executor.start()
        def game_finished(evaluations):
            print('Add {} evaluations to training data.'.format(len(evaluations)))
            training_executor.add_examples(evaluations)

        # Run selfplay to get training data
        print("Start selfplay for epoch {}...".format(epoch))
        if reuse_old_training_data:
            print('Skipping selfplay and use existing data...')
            training_executor._n_test = count_directory_items(os.path.join(current_data_dir, 'test'))
            training_executor._n_training = count_directory_items(os.path.join(current_data_dir, 'training'))
        else:
            parallel_selfplay = distribution.ParallelSelfplayPool(initial_game_state, SimpleNeuralNetwork(),
                                                                  best_model_file, GAMES_PER_EPOCH, game_finished,
                                                                  simulations_per_turn=SIMULATIONS_PER_GAME_TURN,
                                                                  pool_size=8, batch_size=6)
            parallel_selfplay.run()

        # Train a new neural network
        train_file_writer = tf.summary.FileWriter('./tb/graph-{}-train'.format(epoch), tf.get_default_graph())
        print("Start training for epoch {}...".format(epoch))
        for batch in range(1, TRAINING_BATCHES_PER_EPOCH + 1):
            training_executor.run_training_batch(BATCH_SIZE)
            if batch % 50 == 0:
                print('{} batches executed.'.format(batch))
                training_executor.log_training_loss(train_file_writer, batch, batch_size=BATCH_SIZE * 2)
        training_executor.save(current_ckpt_file)
        train_file_writer.close()

        # See if we choose the new one as best network
        print("Start Evaluation versus last epoch...")
        parallel_evaluation = \
            distribution.ParallelSelfplayEvaluationPool(['./simple_8_by_8.map'], SimpleNeuralNetwork(),
                                                        SimpleNeuralNetwork(), best_model_file, current_ckpt_file,
                                                        N_EVALUATION_GAMES,
                                                        simulations_per_turn=SIMULATIONS_PER_GAME_TURN)

        scores = parallel_evaluation.run()
        print('Scores: Old {} vs. New {}'.format(scores[0], scores[1]))

        if scores[1] >= NEEDED_AVG_SCORE:
            print('Using new model, as its average score is {} >= {}'.format(scores[1], NEEDED_AVG_SCORE))
            training_executor.save(best_model_file)
        training_executor.stop()

        if scores[1] >= NEEDED_AVG_SCORE:
            # Evaluate vs. ai trivial
            print("Start AI Evaluation for epoch {}...".format(epoch))
            parallel_tournament = distribution.ParallelAITrivialPool(['./simple_8_by_8.map'], SimpleNeuralNetwork(),
                                                                         best_model_file, N_AI_EVALUATION_GAMES, 1.0)
            scores, stones = parallel_tournament.run()
            print('Tournament Scores: AI {} vs. NN {}'.format(scores[1], scores[0]))
            print('Tournament Stones: AI {} vs. NN {}'.format(stones[1], stones[0]))
            with open(stats_file, 'a') as file:
                file.write('{},{},{},{},{}\n'.format(epoch, scores[0], scores[1], stones[0], stones[1]))


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


def count_directory_items(directory):
    return len([name for name in os.listdir(directory) if os.path.isfile(os.path.join(directory, name))])

class SimpleNeuralNetwork(core.NeuralNetwork):
    def __init__(self):
        super().__init__()

    def construct_network(self):
        self._construct_inputs()

        with tf.name_scope('Convolutional-Layers'):
            conv1 = self._construct_conv_layer(self.one_hot_x, 32, 'cov1')
            res1 = self._construct_residual_block(conv1, 32, 'res1')
            res2 = self._construct_residual_block(res1, 32, 'res2')
            res3 = self._construct_residual_block(res2, 32, 'res3')
            res4 = self._construct_residual_block(res3, 32, 'res4')

        with tf.name_scope('Probability-Head'):
            n_filters = 2

            # Reduce the big amount of convolutional filters to a reasonable size.
            prob_conv = self._construct_conv_layer(res4, n_filters, 'prob_conv', kernel=[1, 1], stride=1)
            # Flattern the output tensor to allow it as input to a fully connected layer.
            flattered_prob_conv = tf.reshape(prob_conv, [-1, n_filters * BOARD_WIDTH * BOARD_HEIGHT])
            # Add a fully connected hidden layer.
            prob_hidden = self._construct_dense_layer(flattered_prob_conv, BOARD_WIDTH * BOARD_HEIGHT, 'prob_hidden',
                                                      activation=tf.nn.leaky_relu)
            prob_hidden_dropout = tf.layers.dropout(prob_hidden, training=self.training)
            # Add a fully connected output layer.
            self.out_prob_logits = self._construct_dense_layer(prob_hidden_dropout, BOARD_WIDTH * BOARD_HEIGHT, 'prob_logits')

            # The final output is a probability distribution and we use the softmax loss.
            # So we need to apply softmax to the output.
            self.out_prob = tf.nn.softmax(self.out_prob_logits)

        with tf.name_scope('Value-Head'):
            # Reduce the big amount of convolutional filters to a reasonable size.
            value_conv = self._construct_conv_layer(res4, 1, 'value_conv', kernel=[1, 1], stride=1)
            # Flattern the output tensor to allow it as input to a fully connected layer.
            flattered_value_conv = tf.reshape(value_conv, [-1, 1 * BOARD_WIDTH * BOARD_HEIGHT])
            # Add a fully connected hidden layer.
            value_hidden = self._construct_dense_layer(flattered_value_conv, BOARD_WIDTH * BOARD_HEIGHT, 'value_hidden',
                                                       activation=tf.nn.leaky_relu)
            value_hidden_dropout = tf.layers.dropout(value_hidden, training=self.training)
            # Add a fully connected output layer.
            value_scalar = self._construct_dense_layer(value_hidden_dropout, 1, 'value_output')

            # Than will give us a value between -1 and 1 as we need it
            self.out_value = tf.nn.tanh(value_scalar)

        with tf.name_scope('Losses'):
            # Value loss is measured in mean square error.
            # Our values are in [-1, 1], so a MSE of 1 would mean that our network simply always outputs the
            # mean of our values. Everything below 1 would be at least a little bit better than guessing.
            self.value_loss = tf.losses.mean_squared_error(self.y_value, self.out_value)

            # Probability loss is the loss of a probability distribution.
            # We have a multilabel problem, where labels are mutually exclusive, but our labels are not
            # one hot, but a target probability distribution.
            # This suggests the softmax cross entropy as an error measure.
            prob_losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.y_prob, logits=self.out_prob_logits)
            self.prob_loss = tf.reduce_mean(prob_losses)

            # Lastly we add L2 regularization
            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self.reg_loss = tf.add_n(reg_losses)

            # The summ of all three are our total loss
            self.loss = tf.add_n([self.prob_loss, self.value_loss, self.reg_loss], name="loss")

        with tf.name_scope('Training'):
            # For now simply go with the 'go-to' adam optimizer.
            optimizer = tf.train.GradientDescentOptimizer(0.1)
            self.training_op = optimizer.minimize(self.loss)

        with tf.name_scope('Logging'):
            self.saver = tf.train.Saver()
            self.init = tf.global_variables_initializer()

            # Log individual losses for debugging.
            self.loss_summary = tf.summary.scalar('loss', self.loss)
            self.value_loss_summary = tf.summary.scalar('value loss', self.value_loss)
            self.prob_loss_summary = tf.summary.scalar('prob loss', self.prob_loss)
            self.reg_loss_summary = tf.summary.scalar('reg loss', self.reg_loss)

    def _construct_inputs(self):
        with tf.name_scope("inputs"):
            # Toggle Flag to enable/disable stuff during training
            self.training = tf.placeholder_with_default(False, shape=(), name='training')
            # Variable to set the 'raw' board input.
            # This means an tensor with shape [batch_size, height, width]
            # where each element is coded as an integer between 0 and N_RAW_VALUES.
            # NOTE: We will not use this for now, as we plan to also pass in the possible
            #       moves directly in the one hot encoded input.
            # self.raw_x = tf.placeholder(tf.uint8, shape=(None, BOARD_HEIGHT, BOARD_WIDTH), name='raw_x')

            # Board will be one hot encoded.
            # Each convolutional input dimension represents one possible stone on a field.
            # self.one_hot_x = tf.one_hot(self.raw_x, N_RAW_VALUES + 1, name='one_hot_x')
            self.one_hot_x = \
                tf.placeholder(FLOAT, shape=(None, BOARD_HEIGHT, BOARD_WIDTH, N_RAW_VALUES + 1), name='one_hot_x')

            # Outputs are the move probabilities for each field and a value estimation for player one.
            # (Note: this is intended to only support two players)
            self.y_prob = tf.placeholder(FLOAT, shape=[None, BOARD_HEIGHT * BOARD_WIDTH], name='y_prob')
            self.y_value = tf.placeholder(FLOAT, shape=[None, 1], name='y_value')

    def _construct_conv_layer(self, input, n_filters, name, kernel=[3, 3], stride=1, normalization=True):
        """Construct a convolutional layer with the given settings.

        Kernel, stride and a optional normalization layer can be configured."""
        with tf.name_scope(name):
            conv = tf.layers.conv2d(
                inputs=input,
                filters=n_filters,
                kernel_size=kernel,
                strides=[stride, stride],
                padding="same",
                activation=tf.nn.leaky_relu,
                kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_LOSS_WEIGHT))
            if normalization:
                return conv

            return tf.layers.batch_normalization(conv, training=self.training)

    def _construct_residual_block(self, input, n_filters, name):
        with tf.name_scope(name):
            conv1 = self._construct_conv_layer(input, n_filters, 'conv1')
            conv1_relu = tf.nn.leaky_relu(conv1)
            conv2 = self._construct_conv_layer(conv1_relu, n_filters, 'conv2')

            skip = input + conv2
            return tf.nn.leaky_relu(skip)


    def _construct_dense_layer(self, input, n_nodes, name, activation=None):
        return tf.layers.dense(inputs=input, units=n_nodes, name=name, activation=activation,
                               kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_LOSS_WEIGHT))

    def log_loss(self, sess, tf_file_writer, evaluations, epoch):
        inputs, prob_outputs, value_outputs = SimpleNeuralNetwork._evaluations_to_input(evaluations)

        # Get all the losses
        prob_loss, value_loss, reg_loss, loss =\
            sess.run([self.prob_loss, self.value_loss, self.reg_loss, self.loss],
                      feed_dict={self.one_hot_x: inputs, self.y_prob: prob_outputs, self.y_value: value_outputs})

        reg_log_summary_str = self.reg_loss_summary.eval(feed_dict={self.reg_loss: reg_loss})
        value_log_summary_str = self.value_loss_summary.eval(feed_dict={self.value_loss: value_loss})
        prob_log_summary_str = self.prob_loss_summary.eval(feed_dict={self.prob_loss: prob_loss})
        log_summary_str = self.loss_summary.eval(feed_dict={self.loss: loss})

        tf_file_writer.add_summary(log_summary_str, epoch)
        tf_file_writer.add_summary(reg_log_summary_str, epoch)
        tf_file_writer.add_summary(value_log_summary_str, epoch)
        tf_file_writer.add_summary(prob_log_summary_str, epoch)

        return loss

    def load_weights(self, sess, filename):
        self.saver.restore(sess, filename)

    def train_batch(self, sess, evaluations):
        inputs, prob_outputs, value_outputs = SimpleNeuralNetwork._evaluations_to_input(evaluations)

        sess.run(self.training_op, feed_dict={self.one_hot_x: inputs, self.y_prob: prob_outputs,
                                              self.y_value: value_outputs, self.training: True})

    def save_weights(self, sess, filename):
        self.saver.save(sess, filename)

    def init_network(self):
        self.init.run()

    def execute_batch(self, sess, evaluations):
        inputs = [SimpleNeuralNetwork._game_sate_to_input(evaluation.game_state, evaluation.possible_moves)
                  for evaluation in evaluations]
        outputs = sess.run([self.out_prob, self.out_value], feed_dict={self.one_hot_x: inputs})

        for i in range(len(evaluations)):
            game_state = evaluations[i].game_state
            height = game_state.board.height
            width = game_state.board.width

            player = game_state.calculate_next_player()
            for y in range(height):
                for x in range(width):
                    evaluations[i].probabilities[(player, (x, y), None)] = outputs[0][i][y * width + x]
            evaluations[i].expected_result[Field.PLAYER_ONE] = outputs[1][i][0]
            evaluations[i].expected_result[Field.PLAYER_TWO] = -outputs[1][i][0]

        return evaluations

    @staticmethod
    def _game_sate_to_input(game_state, possible_moves):
        board = game_state.board

        result = np.empty([board.height, board.width], dtype=int)
        for y in range(board.height):
            for x in range(board.width):
                result[y][x] = Field.to_int8(board[(x, y)])

        # OneHotEncode the inputs
        one_hot_encoder = sklearn.preprocessing.OneHotEncoder(N_RAW_VALUES + 1, sparse=False)
        result = one_hot_encoder.fit_transform(result)
        result = np.reshape(result, [board.height, board.width, N_RAW_VALUES + 1])

        # Mark all possible moves in the last one hot layer
        if not possible_moves:
            next_game_states = game_state.get_next_possible_moves()
            possible_moves = [next_game_state.last_move for next_game_state in next_game_states]

        for possible_move in possible_moves:
            x, y = possible_move[1]
            result[y, x, N_RAW_VALUES] = 1

        return result

    @staticmethod
    def _probabilities_to_output(board, probabilities):
        result = np.zeros([board.height * board.width])

        for move, prob in probabilities.items():
            x, y = move[1]
            result[y * board.width + x] = prob

        return result

    @staticmethod
    def _evaluations_to_input(evaluations):
        normal_evaluations = []
        for evaluation in evaluations:
            normal_evaluations.append(evaluation.convert_to_normal())

        inputs = [SimpleNeuralNetwork._game_sate_to_input(evaluation.game_state, evaluation.possible_moves)
                  for evaluation in normal_evaluations]
        inputs = np.array(inputs)

        value_outputs = np.array([[evaluation.expected_result[Field.PLAYER_ONE]] for evaluation in normal_evaluations])
        prob_outputs = [SimpleNeuralNetwork._probabilities_to_output(evaluation.game_state.board,
                                                                     evaluation.probabilities)
                        for evaluation in normal_evaluations]
        prob_outputs = np.array(prob_outputs)

        return inputs, prob_outputs, value_outputs


if __name__ == '__main__':
    main()
