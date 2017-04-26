# python: 2.7.12
# tensorflow: 0.12.0-rc1

import tensorflow as tf
import inputData
import numpy as np
from params import *

# read data
filename = 'sales001.csv'
trainX, trainY, testX, testY = inputData.getLstmData(filename, input_dimension, output_dimension)
tmpTest = testX.tolist()

# nTest = len(inputX) - nPre
# nVal = len(inputX) - 2*nPre
# trainX, trainY = inputX[:nVal], inputY[:nVal]
# valX, valY = inputX[nVal:nTest], inputY[nVal:nTest]
# testX, testY = inputX[nTest:], inputY[nTest:]

X = tf.placeholder(tf.float32, [batch_size, sequence_len, input_dimension])
Y = tf.placeholder(tf.float32, [batch_size, sequence_len, output_dimension])

# LSTM part
cell = tf.nn.rnn_cell.BasicLSTMCell(num_units)
val, state = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32)

# linear part
W = tf.Variable(tf.truncated_normal([batch_size, output_dimension, output_dimension]))
b = tf.Variable(tf.constant(0.5, shape = [batch_size, output_dimension]))
prediction = tf.matmul(val, W) + b
loss = tf.reduce_mean(tf.abs(Y - prediction) / (Y + prediction))

# optimizer
optimizer1 = tf.train.AdamOptimizer(learning_rate = 0.5)
minimize1 =optimizer1.minimize(loss)
optimizer2 = tf.train.AdamOptimizer(learning_rate = 0.05)
minimize2 =optimizer2.minimize(loss)

# summary
summaries = [tf.scalar_summary('loss', loss), tf.histogram_summary('W', W), tf.histogram_summary('b', b)]
summary_op = tf.merge_summary(summaries)

# error
preY = prediction[0,sequence_len-1, 0]
targetY = Y[0,sequence_len-1, 0]
train_error = tf.abs(targetY - preY) / (targetY + preY)


init_op = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init_op)
summary_writer = tf.train.SummaryWriter('./log', graph_def=sess.graph_def)

epoch = 100
nTrainX = len(trainX)
num_seq = int(nTrainX / sequence_len)
minimize = minimize1
for i in range(epoch):
    ptr = 0
    losses = []
    for j in range(num_seq):
        inp= trainX[ptr:ptr+sequence_len].reshape(batch_size, sequence_len, -1)
        out = trainY[ptr:ptr+sequence_len].reshape(batch_size, sequence_len, -1)
        ptr += sequence_len
        sess.run(minimize, feed_dict = {X: inp, Y: out})
        losses.append(sess.run(loss, feed_dict = {X: inp, Y: out}))
        summary_str = sess.run(summary_op, feed_dict = {X: inp, Y: out})
        summary_writer.add_summary(summary_str, global_step=1)
    m_loss = sess.run(tf.reduce_mean(losses))
    # train error
    inp = trainX[nTrainX-sequence_len:nTrainX].reshape(batch_size, sequence_len, -1)
    out = trainY[nTrainX-sequence_len:nTrainX].reshape(batch_size, sequence_len, -1)
    print("Epoch - ", str(i), m_loss, sess.run(train_error, feed_dict={X: inp, Y: out}))
    # if(m_loss<0.1): minimize = minimize2
    if(m_loss<0.01): break

tePre = []
teLosses = []
for i in range(nPre):
    nTest = len(tmpTest)
    inp = np.array(tmpTest[nTest-sequence_len:nTest]).reshape(batch_size, sequence_len, -1)
    tePre.append(sess.run(prediction, feed_dict={X: inp})[0, nPre-1, 0])
    tmpTest.append([[tePre[i]]])

    # teLosses.append(sess.run(train_error, feed_dict={X: inp, Y: out}))
print('prediction:', tePre)
# print('test loss:', sum(teLosses)/len(teLosses))


sess.close()

tf.split