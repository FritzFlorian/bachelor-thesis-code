import tensorflow as tf
import hometrainer.neural_network as neural_network
import reversialphazero.distributed_8_by_8.input_output_conversion as input_output_conversion


BOARD_SIZE = 12
BOARD_HEIGHT = BOARD_SIZE
BOARD_WIDTH = BOARD_SIZE

# Number of different possible states/contents of a
# single field on the board.
N_RAW_VALUES = 4
FLOAT = tf.float32

L2_LOSS_WEIGHT = 0.002


class SimpleNeuralNetwork(neural_network.NeuralNetwork):
    def input_conversion_function(self):
        return input_output_conversion.input

    def output_conversion_function(self):
        return input_output_conversion.output

    def __init__(self):
        super().__init__()
        self.n_conv_filetrs = 32

    def construct_network(self, sess, graph):
        self._construct_inputs()

        with tf.variable_scope('Convolutional-Layers'):
            conv1 = self._construct_conv_layer(self.one_hot_x, self.n_conv_filetrs, 'cov1', activation=tf.nn.tanh)
            self.conv1_kernel = [v for v in tf.trainable_variables()
                                 if v.name == "Convolutional-Layers/cov1/conv2d/kernel:0"][0]

            res1 = self._construct_residual_block(conv1, self.n_conv_filetrs, 'res1')
            res2 = self._construct_residual_block(res1, self.n_conv_filetrs, 'res2')
            res3 = self._construct_residual_block(res2, self.n_conv_filetrs, 'res3')
            res4 = self._construct_residual_block(res3, self.n_conv_filetrs, 'res4')
            res5 = self._construct_residual_block(res4, self.n_conv_filetrs, 'res5')
            res6 = self._construct_residual_block(res5, self.n_conv_filetrs, 'res6')

        with tf.variable_scope('Probability-Head'):
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

        with tf.variable_scope('Value-Head'):
            n_filters = 1

            # Reduce the big amount of convolutional filters to a reasonable size.
            value_conv = self._construct_conv_layer(res6, n_filters, 'value_conv', kernel=[1, 1], stride=1)
            # Flattern the output tensor to allow it as input to a fully connected layer.
            flattered_value_conv = tf.reshape(value_conv, [-1, n_filters * BOARD_WIDTH * BOARD_HEIGHT])
            # Add a fully connected hidden layer.
            value_hidden = self._construct_dense_layer(flattered_value_conv, BOARD_WIDTH * BOARD_HEIGHT, 'value_hidden',
                                                       activation=tf.nn.tanh)
            value_hidden_dropout = tf.layers.dropout(value_hidden, training=self.training)
            # Add a fully connected output layer.
            value_scalar = self._construct_dense_layer(value_hidden_dropout, 1, 'value_output')

            # Than will give us a value between -1 and 1 as we need it
            self.out_value = tf.nn.tanh(value_scalar)

        with tf.variable_scope('Final-Output'):
            # Combine the output as this is needed to fulfill our internal raw data representation
            self.out_combined = tf.concat([self.out_prob, self.out_value], axis=1)

        with tf.variable_scope('Losses'):
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

        with tf.variable_scope('Training'):
            # Use a simpler optimizer to avoid issues because of it
            optimizer = tf.train.MomentumOptimizer(0.002, 0.9)
            self.training_op = optimizer.minimize(self.loss)

        with tf.variable_scope('Logging'):
            self.saver = tf.train.Saver()
            self.init = tf.global_variables_initializer()

            # Log individual losses for debugging.
            self.loss_summary = tf.summary.scalar('loss', self.loss)
            self.value_loss_summary = tf.summary.scalar('value loss', self.value_loss)
            self.prob_loss_summary = tf.summary.scalar('prob loss', self.prob_loss)
            self.reg_loss_summary = tf.summary.scalar('reg loss', self.reg_loss)

            self.conv1_kernel_summaries = []
            for filter_number in range(self.n_conv_filetrs):
                for image_number in range(N_RAW_VALUES):
                    image = tf.slice(self.conv1_kernel, [0, 0, image_number, filter_number], [3, 3, 1, 1])
                    transposed_image = tf.transpose(image, [3, 0, 1, 2])
                    image_summary = tf.summary.image('conv1-filter-{}-kernel-{}'.format(filter_number, image_number), transposed_image)
                    self.conv1_kernel_summaries.append(image_summary)

    def _construct_inputs(self):
        with tf.variable_scope("inputs"):
            # Toggle Flag to enable/disable stuff during training
            self.training = tf.placeholder_with_default(False, shape=(), name='training')

            # Board will be one hot encoded.
            self.one_hot_x = \
                tf.placeholder(FLOAT, shape=(None, BOARD_HEIGHT, BOARD_WIDTH, N_RAW_VALUES), name='one_hot_x')

            # Concat the expected outputs to one big array, as this is our raw input array
            n_fields = BOARD_HEIGHT * BOARD_WIDTH
            self.y_combined = tf.placeholder(FLOAT, shape=[None, n_fields + 1], name='y_combined')

            # Outputs are the move probabilities for each field and a value estimation for player one.
            # (Note: this is intended to only support two players)
            self.y_prob = tf.slice(self.y_combined, [0, 0], [-1,  n_fields])
            self.y_value = tf.slice(self.y_combined, [0, n_fields], [-1, 1])

    def _construct_conv_layer(self, input, n_filters, name, kernel=[3, 3], stride=1, normalization=True, activation=None):
        """Construct a convolutional layer with the given settings.

        Kernel, stride and a optional normalization layer can be configured."""
        with tf.variable_scope(name):
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
        with tf.variable_scope(name):
            conv1 = self._construct_conv_layer(input, n_filters, 'conv1')
            conv1_relu = tf.nn.tanh(conv1)
            conv2 = self._construct_conv_layer(conv1_relu, n_filters, 'conv2')

            skip = tf.add(input, conv2, 'skip_connection')
            return tf.nn.tanh(skip)

    def _construct_dense_layer(self, input, n_nodes, name, activation=None):
        return tf.layers.dense(input, n_nodes, name=name, activation=activation,
                               kernel_regularizer=tf.contrib.layers.l2_regularizer(L2_LOSS_WEIGHT))

    def log_training_progress(self, sess, tf_file_writer, input_arrays, target_arrays, training_batch):
        # Get all the losses
        prob_loss, value_loss, reg_loss, loss =\
            sess.run([self.prob_loss, self.value_loss, self.reg_loss, self.loss],
                     feed_dict={self.one_hot_x: input_arrays, self.y_combined: target_arrays})

        reg_log_summary_str = self.reg_loss_summary.eval(feed_dict={self.reg_loss: reg_loss})
        value_log_summary_str = self.value_loss_summary.eval(feed_dict={self.value_loss: value_loss})
        prob_log_summary_str = self.prob_loss_summary.eval(feed_dict={self.prob_loss: prob_loss})
        log_summary_str = self.loss_summary.eval(feed_dict={self.loss: loss})

        tf_file_writer.add_summary(log_summary_str, training_batch)
        tf_file_writer.add_summary(reg_log_summary_str, training_batch)
        tf_file_writer.add_summary(value_log_summary_str, training_batch)
        tf_file_writer.add_summary(prob_log_summary_str, training_batch)

        for image_summary in self.conv1_kernel_summaries:
            tf_file_writer.add_summary(image_summary.eval())

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
