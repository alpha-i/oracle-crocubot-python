# Performance comparison: Adam vs SGD
import tensorflow as tf
import numpy as np
import logging

import examples.benchmark_prototype as bench

logger = logging.getLogger('tipper')
logger.addHandler(logging.StreamHandler())
logging.basicConfig(level=logging.DEBUG)

FLAGS = tf.app.flags.FLAGS
N_CYCLES = 1


def run_mnist_tests(optimisation_method):

    accuracy_array = np.zeros(N_CYCLES)
    likeli_array = np.zeros(N_CYCLES)

    tensor_path = '/tmp/'
    train_path = '/tmp/'
    do_2D = True

    for i in range(N_CYCLES):
        accuracy_array[i], metrics = bench.run_mnist_test(train_path, tensor_path, optimisation_method,
                                                          reshape_to_2d=do_2D)
        likeli_array[i] = metrics['log_likelihood_per_sample']

    print(optimisation_method, 'accuracy:', accuracy_array)
    print(optimisation_method, 'likelihoods:', likeli_array)
    print('Mean accuracy:', np.mean(accuracy_array))
    print('Log likelihood:', np.mean(likeli_array))

opt_methods = ['GDO']  # GDO Adam

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
# Adam: [ 0.9796
# GDO:  [ 0.9831  0.982   0.9831  0.9835  0.9827] -> 0.98288

# Results: 200 epoch with gradient clipping AND big extra cost term (40 eval pass, 1 train, lr = 1e-3)
# Repeated to get likelihoods
# GDO accuracy: [ 0.9824  0.9834  0.9834  0.9853  0.9832] -> 0.983
# GDO likelihoods: [-0.06876133 -0.06876963 -0.06817838 -0.063859   -0.06839153] -> -0.0676

# Results: 200 epoch with gradient clipping AND NO big extra cost term (40 eval pass, 1 train, lr = 1e-3)
# GDO accuracy: [0.9787  0.9792  0.9781  0.9793  0.9767]
# GDO likelihoods: [-0.10307571 -0.09865605 -0.10417016 -0.09949302 -0.11244867] -> 0.1035

# Results: 200 epoch with new full extra cost term (40 eval pass, 1 train, lr = 1e-3)
# GDO Accuracy:
#  [0.9829  0.9836  0.9836  0.9821  0.9824  0.9828  0.9827  0.9831  0.9824      0.9836] -> 0.98292
#    likelihoods: [-0.0707785 - 0.06766419 - 0.06848358 - 0.07242996 - 0.0704145 - 0.07105065      - 0.07299981 - 0.07197466 - 0.069243 - 0.06973318]
# -> -0.070477


# Results: 200 epoch no extra cost term but new initialisation (40 eval pass, 1 train, lr = 1e-3)
# GDO [ 0.981   0.9814  0.9808  0.9801  0.9817] -> 0.981
    # Likeli -0.074672919459
# Adam: