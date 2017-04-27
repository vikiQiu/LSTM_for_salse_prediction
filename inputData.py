__author__ = 'VikiQiu'

import numpy as np
from params import *

def getData(filename):
    f = open(filename)
    data = []
    f.readline()
    for line in f.readlines():
        data.append(float(line.strip()))
    f.close()
    return data

def lstm_data(data, sequence_len, output_dimension = 1, labels = False):
    rnn_df = []
    for i in range(len(data) - sequence_len - output_dimension+1):
        if labels:
            rnn_df.append(data[i + sequence_len: i + sequence_len + output_dimension])
        else:
            data_ = data[i: i + sequence_len]
            rnn_df.append(data_ if len(data_) > 1 else [[i] for i in data_])
    return np.array(rnn_df)

def getLstmData(filename, sequence_len, output_dimension = 1, nPre = nPre):
    data = getData(filename)
    X = lstm_data(data, sequence_len, output_dimension, False)
    Y = lstm_data(data, sequence_len, output_dimension, True)
    n = len(X)
    return X[:n-nPre], Y[:n-nPre], X[n-nPre], Y[n-nPre:]

# filename = 'data/sales001.csv'
# X, Y, tX, tY = getLstmData(filename, sequence_len, output_dimension)
# print(X.shape, Y.shape, tX.shape, tY.shape)
# print(X[0:2], Y[0])
# print(tX[len(tX)-1], tY)