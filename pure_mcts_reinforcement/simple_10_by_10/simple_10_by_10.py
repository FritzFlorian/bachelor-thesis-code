import pure_mcts_reinforcement.core as core
import tensorflow as tf
import numpy as np
from reversi.game_core import Field, Board, GameState

BOARD_HEIGHT = 10
BOARD_WIDTH = 10

# Number of different possible states/contents of a
# single field on the board.
N_RAW_VALUES = 3
FLOAT = tf.float32

L2_LOSS_WEIGHT = 1.0


def main():
    with open('simple10by10.map') as file:
        board = Board(file.read())
    initial_game_state = GameState(board)

    # neural_network = SimpleNeuralNetwork()
    # with tf.Graph().as_default():
    #     neural_network.construct_network()
    #     with tf.Session() as sess:
    #         neural_network.init_network()
    #         neural_network.save_weights(sess, './weights.ckpt')

    nn_executor = core.NeuralNetworkExecutor(SimpleNeuralNetwork(), './weights.ckpt')
    nn_executor.start()
    selfplay_executor = core.SelfplayExecutor(initial_game_state, nn_executor, 5)
    training_executor = core.TrainingExecutor(SimpleNeuralNetwork(), './weights.ckpt', 'data')
    training_executor.start()

    for i in range(5):
        print('Start Run {}'.format(i))
        evaluations = selfplay_executor.run()
        print(evaluations)
        print(evaluations[0].probabilities[(Field.PLAYER_ONE, (6, 4), None)])
        training_executor.add_examples(evaluations)

        for j in range(3):
            print('Start Training {}'.format(j))
            training_executor.run_training_batch(32)

    training_executor.save('./weights.ckpt')



class SimpleNeuralNetwork(core.NeuralNetwork):
    def __init__(self):
        super().__init__()

    def construct_network(self):
        self._construct_inputs()

        # Convolutional Part - Should be replaced with proper structure (residual blocks)
        conv1 = self._construct_conv_layer(self.one_hot_x, 32, 'cov1')
        conv1_norm = tf.layers.batch_normalization(conv1, training=self.training)
        conv2 = self._construct_conv_layer(conv1_norm, 32, 'conv2')
        conv2_norm = tf.layers.batch_normalization(conv2, training=self.training)
        conv3 = self._construct_conv_layer(conv2_norm, 32, 'conv3')
        conv3_norm = tf.layers.batch_normalization(conv3, training=self.training)

        # Probability Head
        prob_conv = self._construct_conv_layer(conv3_norm, 4, 'prob_conv',
                                               kernel=[1, 1], stride=1)
        flattend_prob_conv = tf.reshape(prob_conv, [-1, 4 * BOARD_WIDTH * BOARD_HEIGHT])
        # Linear output layer for probabilities
        self.out_prob_logits = tf.layers.dense(inputs=flattend_prob_conv, units=BOARD_WIDTH * BOARD_HEIGHT,
                                               kernel_regularizer=tf.contrib.layers.l1_regularizer(L2_LOSS_WEIGHT))
        self.out_prob = tf.nn.softmax(self.out_prob_logits)

        # Value Head
        value_conv = self._construct_conv_layer(conv3_norm, 1, 'value_conv',
                                               kernel=[1, 1], stride=1)
        value_full = tf.layers.dense(inputs=flattend_prob_conv, units=64, activation=None,
                                     kernel_regularizer=tf.contrib.layers.l1_regularizer(L2_LOSS_WEIGHT))
        value_scalar = tf.layers.dense(inputs=value_full, units=1, activation=None,
                                       kernel_regularizer=tf.contrib.layers.l1_regularizer(L2_LOSS_WEIGHT))
        self.out_value = tf.div(tf.add(tf.nn.tanh(value_scalar), [1]), [2])

        # Losses
        value_loss = tf.losses.mean_squared_error(self.y_value, self.out_value)
        prob_loss = tf.losses.sigmoid_cross_entropy(self.y_prob, self.out_prob_logits)
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)

        self.loss = tf.add_n([value_loss, prob_loss] + reg_losses, name="loss")

        # Training Operation
        optimizer = tf.train.AdamOptimizer(learning_rate=0.002, beta1=0.8, beta2=0.995)
        self.training_op = optimizer.minimize(self.loss)

        # Housekeeping
        self.saver = tf.train.Saver()
        self.init = tf.global_variables_initializer()
        self.loss_summary = tf.summary.scalar('loss', self.loss)

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
                kernel_regularizer=tf.contrib.layers.l1_regularizer(L2_LOSS_WEIGHT))
            return conv

    def log_loss(self, tf_file_writer, evaluations, epoch):
        inputs, prob_outputs, value_outputs = SimpleNeuralNetwork._evaluations_to_input(evaluations)

        loss = self.loss.eval(feed_dict={self.raw_x: inputs, self.out_prob: prob_outputs,
                                         self.out_value: value_outputs})
        summary_str = self.loss_summary.eval(feed_dict={loss: loss})
        tf_file_writer.add_summary(summary_str, epoch)

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

    def execute_batch(self, sess, game_states):
        evaluations = [core.Evaluation(game_state) for game_state in game_states]
        for evaluation in evaluations:
            evaluation.convert_to_normal()

        inputs = [SimpleNeuralNetwork._game_board_to_input(evaluation.game_state.board) for evaluation in evaluations]
        outputs = sess.run([self.out_prob, self.out_value], feed_dict={self.raw_x: inputs})

        for i in range(len(evaluations)):
            game_state = game_states[i]
            height = game_state.board.height
            width = game_state.board.width
            # TODO: It's just horrible to execute the move EACH time, just to find the next player...
            (player, _, _) = game_state.get_next_possible_moves()[0].last_move
            for y in range(height):
                for x in range(width):
                    evaluations[i].probabilities[(player, (x, y), None)] = outputs[0][i][y * width + x]
            evaluations[i].expected_result[Field.PLAYER_ONE] = outputs[1][i][0]
            evaluations[i].expected_result[Field.PLAYER_TWO] = 1.0 - outputs[1][i][0]

        for evaluation in evaluations:
            evaluation.convert_from_normal()

        return evaluations

    @staticmethod
    def _game_board_to_input(board):
        result = np.empty([board.height, board.width], dtype=int)
        for y in range(board.height):
            for x in range(board.width):
                result[y][x] = SimpleNeuralNetwork._field_to_int(board[(x, y)])

        return result

    @staticmethod
    def _field_to_int(field):
        if field == Field.EMPTY:
            return 0
        if field == Field.PLAYER_ONE:
            return 1
        if field == Field.PLAYER_TWO:
            return 2

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
