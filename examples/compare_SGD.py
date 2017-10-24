# Performance comparison: Adam vs SGD
import tensorflow as tf
import numpy as np
import argparse

import examples.benchmark_prototype as bench
FLAGS = tf.app.flags.FLAGS

N_CYCLES = 5


def run_mnist_tests(method):

    accuracy_array = np.zeros(N_CYCLES)

    tensor_path = '/tmp/'
    train_path = '/tmp/'

    for i in range(N_CYCLES):
        accuracy_array[i] = bench.run_mnist_test(train_path, tensor_path, method)

    print(method, 'accuracy:', accuracy_array)
    print('Mean accuracy:', np.mean(accuracy_array))


opt_methods = ['GDO']  # GDO Adam

for method in opt_methods:
    run_mnist_tests(method)


# Results: 20 epoch
# Adam: [ 0.9216  0.9214  0.9189  0.9232  0.9224] -> mean 0.9215
# SGD:  [ 0.9244  0.9243  0.9263  0.9243  0.9249] -> mean 0.9248

# Results: 200 epoch
# Adam accuracy: [ 0.9714  0.9709  0.9701  0.9725  0.9731]
# SGD:  [ 0.9703  0.968   0.9691  0.969   0.9694] ->  Mean accuracy: 0.9691