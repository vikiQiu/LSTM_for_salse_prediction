__author__ = 'VikiQiu'
# https://github.com/aymericdamien/TensorFlow-Examples/blob/0.11/examples/3_NeuralNetworks/recurrent_network.py
# https://github.com/vikiQiu/LSTM_for_salse_prediction

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import inputData

# Parameters
learning_rate = 0.1
training_iters = 10000
batch_size = 1
display_step = 10

# Network Parameters
input_dimension = 1 # MNIST data input (img shape: 28*28)
sequence_len = 5 # timesteps
n_hidden = 128 # hidden layer num of features
output_dimension = 10 # MNIST total classes (0-9 digits)

# read data
filename = 'sales001.csv'
trainX, trainY, testX, testY = inputData.getLstmData(filename, sequence_len, output_dimension)
tmpTest = testX.tolist()

# tf Graph input
x = tf.placeholder("float", [None, sequence_len, input_dimension])
y = tf.placeholder("float", [None, output_dimension])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, output_dimension]))
}
biases = {
    'out': tf.Variable(tf.random_normal([output_dimension]))
}


def RNN(x, weights, biases):
    x = tf.unstack(x, sequence_len, 1)
    # Define a lstm cell with tensorflow
    lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)

    # Get lstm cell output
    outputs, states = rnn.rnn(lstm_cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    return tf.matmul(outputs[-1], weights['out']) + biases['out']

pred = RNN(x, weights, biases)

# Define loss and optimizer
cost = tf.reduce_mean(tf.abs(y - pred) / (y + pred))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
accuracy = cost

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)
    step = 0
    nTrain = len(trainX)
    # Keep training until reach max iterations
    while step * batch_size < nTrain:
        endInd = min(nTrain, (step+1)*batch_size)
        batch_x = trainX[step*batch_size:endInd].reshape(-1, sequence_len, input_dimension)
        batch_y = trainY[step*batch_size:endInd]
        # Run optimization op (backprop)
        sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
        if step % display_step == 0:
            # Calculate batch accuracy
            acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # Calculate batch loss
            loss = sess.run(cost, feed_dict={x: batch_x, y: batch_y})
            print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                  "{:.6f}".format(loss) + ", Training Accuracy= " + \
                  "{:.5f}".format(acc))
        step += 1
    print("Optimization Finished!")

    # Calculate accuracy for 128 mnist test images
    # test_len = 128
    # test_data = mnist.test.images[:test_len].reshape((-1, sequence_len, input_dimension))
    # test_label = mnist.test.labels[:test_len]
    # print("Testing Accuracy:", \
    #     sess.run(accuracy, feed_dict={x: test_data, y: test_label}))
