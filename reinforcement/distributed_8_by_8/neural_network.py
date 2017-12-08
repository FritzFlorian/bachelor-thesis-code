import tensorflow as tf
import reinforcement.neural_network as neural_network
import reinforcement.distributed_8_by_8.input_output_conversion as input_output_conversion


BOARD_HEIGHT = 8
BOARD_WIDTH = 8

# Number of different possible states/contents of a
# single field on the board.
N_RAW_VALUES = 3
FLOAT = tf.float32

L2_LOSS_WEIGHT = 0.001


class SimpleNeuralNetwork(neural_network.NeuralNetwork):
    def input_conversion_function(self):
        return input_output_conversion.input

    def output_conversion_function(self):
        return input_output_conversion.output

    def __init__(self):
        super().__init__()

    def construct_network(self):
        self._construct_inputs()

        with tf.name_scope('Convolutional-Layers'):
            conv1 = self._construct_conv_layer(self.one_hot_x, 32, 'cov1', activation=tf.nn.tanh)
            res1 = self._construct_residual_block(conv1, 32, 'res1')
            res2 = self._construct_residual_block(res1, 32, 'res2')
            res3 = self._construct_residual_block(res2, 32, 'res3')
            res4 = self._construct_residual_block(res3, 32, 'res4')
            res5 = self._construct_residual_block(res4, 32, 'res5')
            res6 = self._construct_residual_block(res5, 32, 'res6')

        with tf.name_scope('Probability-Head'):
            n_filters = 2

            # Reduce the big amount of convolutional filters to a reasonable size.
            prob_conv = self._construct_conv_layer(res6, n_filters, 'prob_conv', kernel=[1, 1], stride=1)
            # Flattern the output tensor to allow it as input to a fully connected layer.
            flattered_prob_conv = tf.reshape(prob_conv, [-1, n_filters * BOARD_WIDTH * BOARD_HEIGHT])
            # Add a fully connected hidden layer.
            prob_hidden = self._construct_dense_layer(flattered_prob_conv, BOARD_WIDTH * BOARD_HEIGHT, 'prob_hidden',
                                                      activation=tf.nn.tanh)
            prob_hidden_dropout = tf.layers.dropout(prob_hidden, training=self.training)
            # Add a fully connected output layer.
            self.out_prob_logits = self._construct_dense_layer(prob_hidden_dropout, BOARD_WIDTH * BOARD_HEIGHT, 'prob_logits')

            # The final output is a probability distribution and we use the softmax loss.
            # So we need to apply softmax to the output.
            self.out_prob = tf.nn.softmax(self.out_prob_logits)

        with tf.name_scope('Value-Head'):
            # Reduce the big amount of convolutional filters to a reasonable size.
            value_conv = self._construct_conv_layer(res6, 1, 'value_conv', kernel=[1, 1], stride=1)
            # Flattern the output tensor to allow it as input to a fully connected layer.
            flattered_value_conv = tf.reshape(value_conv, [-1, 1 * BOARD_WIDTH * BOARD_HEIGHT])
            # Add a fully connected hidden layer.
            value_hidden = self._construct_dense_layer(flattered_value_conv, BOARD_WIDTH * BOARD_HEIGHT, 'value_hidden',
                                                       activation=tf.nn.tanh)
            value_hidden_dropout = tf.layers.dropout(value_hidden, training=self.training)
            # Add a fully connected output layer.
            value_scalar = self._construct_dense_layer(value_hidden_dropout, 1, 'value_output')

            # Than will give us a value between -1 and 1 as we need it
            self.out_value = tf.nn.tanh(value_scalar)

        with tf.name_scope('Final-Output'):
            # Combine the output as this is needed to fulfill our internal raw data representation
            self.out_combined = tf.concat([self.out_prob, self.out_value], axis=1)

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
            # Use a simpler optimizer to avoid issues because of it
            optimizer = tf.train.MomentumOptimizer(0.001, 0.9)
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

            # Concat the outputs to one big array, as this is our raw input array
            self.y_combined = tf.concat([self.y_prob, self.y_value], axis=1)

    def _construct_conv_layer(self, input, n_filters, name, kernel=[3, 3], stride=1, normalization=True, activation=None):
        """Construct a convolutional layer with the given settings.

        Kernel, stride and a optional normalization layer can be configured."""
        with tf.name_scope(name):
            conv = tf.layers.conv2d(
                inputs=input,
                filters=n_filters,
                kernel_size=kernel,
                strides=[stride, stride],
                padding="same",
                activation=activation,
                kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_LOSS_WEIGHT))
            if normalization:
                return conv

            return tf.layers.batch_normalization(conv, training=self.training)

    def _construct_residual_block(self, input, n_filters, name):
        with tf.name_scope(name):
            conv1 = self._construct_conv_layer(input, n_filters, 'conv1')
            conv1_relu = tf.nn.tanh(conv1)
            conv2 = self._construct_conv_layer(conv1_relu, n_filters, 'conv2')

            skip = tf.add(input, conv2, 'skip_connection')
            return tf.nn.tanh(skip)

    def _construct_dense_layer(self, input, n_nodes, name, activation=None):
        return tf.layers.dense(inputs=input, units=n_nodes, name=name, activation=activation,
                               kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_LOSS_WEIGHT))

    def log_loss(self, sess, tf_file_writer, input_arrays, target_arrays, epoch):
        # Get all the losses
        prob_loss, value_loss, reg_loss, loss =\
            sess.run([self.prob_loss, self.value_loss, self.reg_loss, self.loss],
                     feed_dict={self.one_hot_x: input_arrays, self.y_combined: target_arrays})

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

    def train_batch(self, sess, input_arrays, target_arrays):
        sess.run(self.training_op, feed_dict={self.one_hot_x: input_arrays, self.y_combined: target_arrays,
                                              self.training: True})

    def save_weights(self, sess, filename):
        self.saver.save(sess, filename)

    def init_network(self):
        self.init.run()

    def execute_batch(self, sess, input_arrays):
        return sess.run(self.out_combined, feed_dict={self.one_hot_x: input_arrays})
