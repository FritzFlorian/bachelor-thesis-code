import os
import pandas as pd
import tensorflow as tf
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split

DATA_ROOT = "training_data"
X_PATH = os.path.join(DATA_ROOT, "x_values.csv")
Y_PATH = os.path.join(DATA_ROOT, "y_values.csv")

now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
root_logdir = "tf_output"
logdir_train = "{}/run-{}-train/".format(root_logdir, now)
logdir_test = "{}/run-{}-test/".format(root_logdir, now)

BOARD_HEIGHT = 12
BOARD_WIDTH = 12


##############################
# Load datasets
##############################

# load and reshape the data to a list of 2D arrays
def load_12_by_12_data(x_path=X_PATH, y_path=Y_PATH):
    x, y = np.genfromtxt(x_path, delimiter=","), np.genfromtxt(y_path, delimiter=",")
    x_reshaped = np.reshape(x, (len(x), BOARD_HEIGHT, BOARD_WIDTH))
    return x_reshaped, y 


# Prepare training and test sets
x, y = load_12_by_12_data()
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=0.2, random_state=42)

train_size = len(train_x)
print("Loaded {0} datasets, split into {1} training and {2} test rows."
        .format(len(x), train_size, len(test_x)))

BATCH_SIZE = 100
train_x_batches = np.array_split(train_x, BATCH_SIZE)
train_y_batches = np.array_split(train_y, BATCH_SIZE)
num_batches = len(train_x_batches)

##############################
# Create the tensorflow graph
##############################
with tf.name_scope("inputs"):
    training = tf.placeholder_with_default(False, shape=(), name='training')
    X = tf.placeholder(tf.uint8, shape=(None, BOARD_HEIGHT, BOARD_WIDTH), name="X")
    x_one_hot = tf.one_hot(X, 3, name="x_one_hot")
    y = tf.placeholder(tf.int32, shape=[None, BOARD_HEIGHT * BOARD_WIDTH], name="y")

# Convolutional Layer 1
with tf.name_scope("conv1"):
    conv1 = tf.layers.conv2d(
            inputs=x_one_hot,
            filters=16,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu)

# Convolutional Layer 2
with tf.name_scope("conv2"):
    conv2 = tf.layers.conv2d(
            inputs=conv1,
            filters=32,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu)

# Convolutional Layer 3
with tf.name_scope("conv3"):
    conv3 = tf.layers.conv2d(
            inputs=conv2,
            filters=64,
            kernel_size=[3, 3],
            padding="same",
            activation=tf.nn.relu)

# Dense Layer
with tf.name_scope("dense1"):
    flattend_last_layer = tf.reshape(conv3, [-1, 64 * BOARD_WIDTH * BOARD_HEIGHT])
    dense = tf.layers.dense(inputs=flattend_last_layer, units=2048, activation=tf.nn.relu)
    dropout = tf.layers.dropout(inputs=dense, rate=0.6, training=training)

# Output Layer
with tf.name_scope("output"):
    output = tf.layers.dense(inputs=dropout, units=BOARD_WIDTH * BOARD_HEIGHT)

with tf.name_scope("loss"):
    loss = tf.losses.mean_squared_error(output, y)

with tf.name_scope("train"):
    optimizer = tf.train.AdamOptimizer(learning_rate=0.002, beta1=0.8, beta2=0.995)
    training_op = optimizer.minimize(loss)

# Logging/Saving/Init
with tf.name_scope("init_and_save"):
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    
loss_summary = tf.summary.scalar('loss', loss)
test_loss_summary = tf.summary.scalar('test loss', loss)
test_file_writer = tf.summary.FileWriter(logdir_test, tf.get_default_graph())
train_file_writer = tf.summary.FileWriter(logdir_train, tf.get_default_graph())

##############################
# Run the graph for training
##############################
n_epochs = 30 

with tf.Session() as sess:
    init.run()
    # TODO: saver.restore(sess, "path_to_checkpoint")

    for epoch in range(n_epochs):
        for i in range(num_batches):
            x_batch = train_x_batches[i]
            y_batch = train_y_batches[i]
            sess.run(training_op, feed_dict={X: x_batch, y: y_batch, training: True})

        loss_train = loss.eval(feed_dict={X: train_x_batches[0], y: train_y_batches[0]})
        loss_test = loss.eval(feed_dict={X: test_x, y: test_y})
        print(epoch, "Train loss:", loss_train, "Test loss:", loss_test)

        summary_str_train = loss_summary.eval(feed_dict={loss: loss_train})
        summary_str_test = loss_summary.eval(feed_dict={loss: loss_test})
        train_file_writer.add_summary(summary_str_train, epoch)
        test_file_writer.add_summary(summary_str_test, epoch)

        # During Training, keep regular checkpoints
        saver.save(sess, "./checkpoint.ckpt")



    # After training is finished
    saver.save(sess, "./final.ckpt")
    manual_test = output.eval(feed_dict={X: [test_x[0]], y: [test_y[0]]})
    print(manual_test)
    print(test_y[0])
    out_array = [manual_test[0], test_y[0]]
    np.savetxt("test.csv", out_array, delimiter=",", fmt="%10.8f")

test_file_writer.close()
train_file_writer.close()
