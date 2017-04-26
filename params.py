__author__ = 'VikiQiu'

""" arguments:
  num_units: The number of units in the LSTM cell.
  num_layer: The number of LSTM layers.
"""

num_layers = 2
is_training = True
# keep_prob = 1
batch_size = 1
sequence_len = 30
input_dimension = 1
output_dimension = 1
num_units = output_dimension
nPre = 7
# time_steps = 1