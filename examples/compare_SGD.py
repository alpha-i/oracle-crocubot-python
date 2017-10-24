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


opt_methods = ['GDO', 'Adam']  # GDO Adam

for method in opt_methods:
    run_mnist_tests(method)


# Results: 20 epoch
# Adam: [ 0.9216  0.9214  0.9189  0.9232  0.9224] -> mean 0.9215
# GDO:  [ 0.9244  0.9243  0.9263  0.9243  0.9249] -> mean 0.9248

# Results: 200 epoch
# Adam accuracy: [ 0.9714  0.9709  0.9701  0.9725  0.9731]
# GDO:  [ 0.9703  0.968   0.9691  0.969   0.9694] ->  Mean accuracy: 0.9691

# Results: 20 epoch with gradient clipping (40 eval pass, 1 train, lr = 2e-3)
# Adam: [ 0.9753  0.9768  0.9771  0.977   0.9771] -> Mean 0.97666
# GDO:  0.9797  0.981   0.9825  0.9812  0.9828] -> Mean 0.98144

# Results: 20 epoch with gradient clipping AND extra cost term (40 eval pass, 1 train, lr = 3e-3)
# Adam:  0.9772  0.9768  0.9718  0.9794  0.978  -> Mean 0.97664
# GDO: 0.9778  0.9816  0.9798  0.9801  0.981 ] -> Mean 0.98006

# Results: 20 epoch with gradient clipping AND big extra cost term (40 eval pass, 1 train, lr = 3e-3)
# Adam:  [ 0.9767  0.9733  0.9737  0.9748  0.9766] -> 0.97502
# GDO: [ 0.9825  0.9835  0.9828  0.9838  0.9824] -> 0.983  ****

# Results: 200 epoch with gradient clipping AND big extra cost term (40 eval pass, 1 train, lr = 1e-3)
# Adam: [
# GDO:  [ 0.9831  0.982   0.9831  0.9835  0.9827] -> 0.98288

