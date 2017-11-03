# Used for retrieving non-financial data, and saving/retrieving non-financial models
# Will not be used by quant workflow
# TODO the comment above is not true cause this module is imported in the crocubot.train module.
# TODO move this iotools somewhere in the benchmark namespace

import os
import tensorflow as tf

from alphai_data_sources.generator import BatchGenerator

FLAGS = tf.app.flags.FLAGS
batch_generator = BatchGenerator()


def reset_mnist():
    batch_generator.reset_mnist()


def build_check_point_filename(series_name, topology):
    """ File used for storing the network parameters.

    :param str series_name: Identify the data on which the network was trained: MNIST, low_noise, randomwalk, etc
    :param Topology topology: Info on network shape
    :return:
    """

    depth_string = str(topology.n_layers)
    breadth_string = str(topology.n_features_per_series)
    series_string = str(topology.n_series)

    bitstring = str(FLAGS.TF_TYPE)
    path = FLAGS.model_save_path

    file_name = "{}model_{}_{}_{}x{}.ckpt".format(bitstring[-2:], series_name, series_string, depth_string,
                                                  breadth_string)
    return os.path.join(path, file_name)
