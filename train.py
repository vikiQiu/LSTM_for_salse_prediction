__author__ = 'VikiQiu'
# https://github.com/aymericdamien/TensorFlow-Examples/blob/0.11/examples/3_NeuralNetworks/recurrent_network.py
# https://github.com/vikiQiu/LSTM_for_salse_prediction

import tensorflow as tf
from tensorflow.python.ops import rnn, rnn_cell
import inputData
import numpy as np

# Parameters
learning_rate = 0.1
training_iters = 10000
batch_size = 1
display_step = 10
nPre = 7

# Network Parameters
input_dimension = 1 # univariate time series
sequence_len = 10 # timesteps
n_hidden = 128 # hidden layer num of features
output_dimension = 7 # predict 7 sales very time
num_layer = 2

# read data
fileInd = 1
filename = '../../SparkRSession1/data/input/sales'+str(fileInd)+'.csv'
trainX, trainY, testX, testY = inputData.getLstmData(filename, sequence_len, output_dimension)
tmpTest = testX.tolist()

# tf Graph input
x = tf.placeholder("float", [None, sequence_len, input_dimension])
y = tf.placeholder("float", [None, output_dimension])

# Define weights
weights = {
    'out': tf.Variable(tf.random_normal([n_hidden, output_dimension])),
    'out2': tf.Variable(tf.random_normal([output_dimension, output_dimension]))
}
biases = {
    'out': tf.Variable(tf.random_normal([output_dimension])),
    'out2': tf.Variable(tf.random_normal([output_dimension]))
}


def RNN(x, weights, biases,  num_layer = num_layer):
    x = tf.unstack(x, sequence_len, 1)
    # Define a lstm cell with tensorflow
    lstm_cell = rnn_cell.BasicLSTMCell(n_hidden, forget_bias=1.0)
    cell = tf.nn.rnn_cell.MultiRNNCell([lstm_cell]*num_layer)

    # Get lstm cell output
    outputs, states = rnn.rnn(cell, x, dtype=tf.float32)

    # Linear activation, using rnn inner loop last output
    h = tf.matmul(outputs[-1], weights['out']) + biases['out']
    # return tf.matmul(h, weights['out2']) + biases['out2']
    return  h

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
    nTrain = len(trainX)
    epoch = 2
    for i in range(epoch) :
        step = 0
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
                print("ShopID:"+str(fileInd)+"Iter " + str(i)+","+str(step) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
            step += 1
    print("Optimization Finished!")

    # output_dimension = 7
    inp = testX.reshape(-1, sequence_len, input_dimension)
    out = testY[len(testY)-1].reshape(-1, output_dimension)
    tePre = sess.run(pred, feed_dict={x: inp})
    loss = sess.run(accuracy, feed_dict={x: inp, y: out})
    print("seq_len="+str(sequence_len)+",loss="+str(loss))
    print(tePre)

    # f = open('data/lstmRes.txt', 'w')
    # f.writelines(tePre[0])
    # f.close()
    np.savetxt('data/lstmRes.txt', tePre)
