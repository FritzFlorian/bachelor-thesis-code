import pure_mcts_reinforcement.core as core
import pure_mcts_reinforcement.distribution as distribution
import tensorflow as tf
import numpy as np
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

L1_LOSS_WEIGHT = 0.00001

# Number of games played to gather training data per epoch (per NN configuration)
GAMES_PER_EPOCH = 16
SIMULATIONS_PER_GAME_TURN = 20

TRAINING_BATCHES_PER_EPOCH = 2_000
BATCH_SIZE = 32

N_EPOCHS = 10

CHECKPOINT_FOLDER = './checkpoints'
DATA_FOLDER = './data'

N_EVALUATION_GAMES = 16
NEEDED_AVG_SCORE = 0.55

N_AI_EVALUATION_GAMES = 8


def main():
    # FIXME: Tensorflow issue when using multiprocesning
    multiprocessing.set_start_method('spawn', True)

    with open('simple_8_by_8.map') as file:
        board = Board(file.read())
    initial_game_state = GameState(board)

    best_model_dir = os.path.join(CHECKPOINT_FOLDER, 'current')
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

    run_dir = 'run_{}'.format(round(time.time() * 1000))
    create_directory(os.path.join(CHECKPOINT_FOLDER, run_dir))
    stats_file = os.path.join(CHECKPOINT_FOLDER, run_dir, 'stats.csv')
    with open(stats_file, 'w') as file:
        file.write('epoch,nn_score,ai_score,nn_stones,ai_stones\n')

    for epoch in range(N_EPOCHS):
        current_data_dir = os.path.join(DATA_FOLDER, run_dir, 'epoch-{0:05d}'.format(epoch))
        current_ckpt_dir = os.path.join(CHECKPOINT_FOLDER, run_dir, 'ckpt-{0:05d}'.format(epoch))
        current_ckpt_file = os.path.join(current_ckpt_dir, 'checkpoint.ckpt')
        create_directory(current_ckpt_dir)

        training_executor = core.TrainingExecutor(SimpleNeuralNetwork(), best_model_file, current_data_dir)
        training_executor.start()
        def game_finished(evaluations):
            for evaluation in evaluations:
                # Try to prevent overfitting a little
                if random.choice([True, False]):
                    evaluation.mirror_vertical()
            print('Add {} evaluations to training data.'.format(len(evaluations)))
            training_executor.add_examples(evaluations)

        # Run selfplay to get training data
        print("Start selfplay for epoch {}...".format(epoch))
        parallel_selfplay = distribution.ParallelSelfplayPool(initial_game_state, SimpleNeuralNetwork(),
                                                              best_model_file, GAMES_PER_EPOCH, game_finished,
                                                              simulations_per_turn=SIMULATIONS_PER_GAME_TURN)
        parallel_selfplay.run()

        # Train a new neural network
        test_file_writer = tf.summary.FileWriter('./tb/graph-{}-test'.format(epoch), tf.get_default_graph())
        train_file_writer = tf.summary.FileWriter('./tb/graph-{}-train'.format(epoch), tf.get_default_graph())
        print("Start training for epoch {}...".format(epoch))
        for batch in range(TRAINING_BATCHES_PER_EPOCH):
            training_executor.run_training_batch(BATCH_SIZE)
            if batch % 50 == 0:
                print('{} batches executed.'.format(batch))
                training_executor.log_test_loss(test_file_writer, batch)
                training_executor.log_training_loss(train_file_writer, batch)
        training_executor.save(current_ckpt_file)
        test_file_writer.close()
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
                                                                         best_model_file, N_EVALUATION_GAMES, 1.0)
            scores, stones = parallel_tournament.run()
            print('Tournament Scores: AI {} vs. NN {}'.format(scores[1], scores[0]))
            print('Tournament Stones: AI {} vs. NN {}'.format(stones[1], stones[0]))
            with open(stats_file, 'a') as file:
                file.write('{},{},{},{},{}\n'.format(epoch, scores[0], scores[1], stones[0], stones[1]))


def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)


class SimpleNeuralNetwork(core.NeuralNetwork):
    def __init__(self):
        super().__init__()

    def construct_network(self):
        self._construct_inputs()

        # Convolutional Part - Should be replaced with proper structure (residual blocks)
        conv1 = self._construct_conv_layer(self.one_hot_x, 48, 'cov1')
        conv1_norm = tf.layers.batch_normalization(conv1, training=self.training)
        conv2 = self._construct_conv_layer(conv1_norm, 48, 'conv2')
        conv2_norm = tf.layers.batch_normalization(conv2, training=self.training)
        conv3 = self._construct_conv_layer(conv2_norm, 48, 'conv3')
        conv3_norm = tf.layers.batch_normalization(conv3, training=self.training)
        conv4 = self._construct_conv_layer(conv3_norm, 48, 'conv4')
        conv4_norm = tf.layers.batch_normalization(conv4, training=self.training)
        conv5 = self._construct_conv_layer(conv4_norm, 48, 'conv5')
        conv5_norm = tf.layers.batch_normalization(conv5, training=self.training)

        # Probability Head
        prob_conv = self._construct_conv_layer(conv5_norm, 4, 'prob_conv',
                                               kernel=[1, 1], stride=1)
        flattend_prob_conv = tf.reshape(prob_conv, [-1, 4 * BOARD_WIDTH * BOARD_HEIGHT])
        # Linear output layer for probabilities
        self.out_prob_logits = tf.layers.dense(inputs=flattend_prob_conv, units=BOARD_WIDTH * BOARD_HEIGHT,
                                               kernel_regularizer=tf.contrib.layers.l1_regularizer(L1_LOSS_WEIGHT))
        self.out_prob = tf.nn.softmax(self.out_prob_logits)

        # Value Head
        value_conv = self._construct_conv_layer(conv3_norm, 1, 'value_conv',
                                               kernel=[1, 1], stride=1)
        value_full = tf.layers.dense(inputs=flattend_prob_conv, units=64, activation=None,
                                     kernel_regularizer=tf.contrib.layers.l1_regularizer(L1_LOSS_WEIGHT))
        value_scalar = tf.layers.dense(inputs=value_full, units=1, activation=None,
                                       kernel_regularizer=tf.contrib.layers.l1_regularizer(L1_LOSS_WEIGHT))
        self.out_value = tf.div(tf.add(tf.nn.tanh(value_scalar), [1]), [2])

        # Losses
        self.value_loss = tf.losses.mean_squared_error(self.y_value, self.out_value)
        self.prob_loss = tf.losses.sigmoid_cross_entropy(self.y_prob, self.out_prob_logits)
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        self.reg_loss = tf.add_n(reg_losses)

        self.loss = tf.add_n([self.value_loss, self.prob_loss, self.reg_loss], name="loss")

        # Training Operation
        optimizer = tf.train.AdamOptimizer(learning_rate=0.002, beta1=0.8, beta2=0.995)
        self.training_op = optimizer.minimize(self.loss)

        # Housekeeping
        self.saver = tf.train.Saver()
        self.init = tf.global_variables_initializer()
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
            self.raw_x = tf.placeholder(tf.uint8, shape=(None, BOARD_HEIGHT, BOARD_WIDTH), name='raw_x')

            # Board will be one hot encoded.
            # Each convolutional input dimension represents one possible stone on a field.
            self.one_hot_x = tf.one_hot(self.raw_x, N_RAW_VALUES, name='one_hot_x')

            # Outputs are the move probabilities for each field and a value estimation for player one.
            # (Note: this is intended to only support two players)
            self.y_prob = tf.placeholder(FLOAT, shape=[None, BOARD_HEIGHT * BOARD_WIDTH], name='y_prob')
            self.y_value = tf.placeholder(FLOAT, shape=[None, 1], name='y_value')

    def _construct_conv_layer(self, input, n_filters, name, kernel=[3, 3], stride=1):
        with tf.name_scope(name):
            conv = tf.layers.conv2d(
                inputs=input,
                filters=n_filters,
                kernel_size=kernel,
                strides=[stride, stride],
                padding="same",
                activation=tf.nn.relu,
                kernel_regularizer=tf.contrib.layers.l1_regularizer(L1_LOSS_WEIGHT))
            return conv

    def log_loss(self, sess, tf_file_writer, evaluations, epoch):
        inputs, prob_outputs, value_outputs = SimpleNeuralNetwork._evaluations_to_input(evaluations)

        # Get all the losses
        prob_loss, value_loss, reg_loss, loss =\
            sess.run([self.prob_loss, self.value_loss, self.reg_loss, self.loss],
                      feed_dict={self.raw_x: inputs, self.y_prob: prob_outputs, self.y_value: value_outputs})

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

        sess.run(self.training_op, feed_dict={self.raw_x: inputs, self.y_prob: prob_outputs,
                                              self.y_value: value_outputs, self.training: True})

    def save_weights(self, sess, filename):
        self.saver.save(sess, filename)

    def init_network(self):
        self.init.run()

    def execute_batch(self, sess, evaluations):
        inputs = [SimpleNeuralNetwork._game_board_to_input(evaluation.game_state.board) for evaluation in evaluations]
        outputs = sess.run([self.out_prob, self.out_value], feed_dict={self.raw_x: inputs})

        for i in range(len(evaluations)):
            game_state = evaluations[i].game_state
            height = game_state.board.height
            width = game_state.board.width

            player = game_state.calculate_next_player()
            for y in range(height):
                for x in range(width):
                    evaluations[i].probabilities[(player, (x, y), None)] = outputs[0][i][y * width + x]
            evaluations[i].expected_result[Field.PLAYER_ONE] = outputs[1][i][0]
            evaluations[i].expected_result[Field.PLAYER_TWO] = 1.0 - outputs[1][i][0]

        return evaluations

    @staticmethod
    def _game_board_to_input(board):
        result = np.empty([board.height, board.width], dtype=int)
        for y in range(board.height):
            for x in range(board.width):
                result[y][x] = Field.to_int8(board[(x, y)])

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
        for evaluation in evaluations:
            evaluation.convert_to_normal()

        inputs = [SimpleNeuralNetwork._game_board_to_input(evaluation.game_state.board) for evaluation in evaluations]
        inputs = np.array(inputs)

        value_outputs = np.array([[evaluation.expected_result[Field.PLAYER_ONE]] for evaluation in evaluations])
        prob_outputs = [SimpleNeuralNetwork._probabilities_to_output(evaluation.game_state.board,
                                                                     evaluation.probabilities)
                        for evaluation in evaluations]
        prob_outputs = np.array(prob_outputs)

        return inputs, prob_outputs, value_outputs


if __name__ == '__main__':
    main()
