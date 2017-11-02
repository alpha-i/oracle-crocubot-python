# Used for retrieving non-financial data, and saving/retrieving non-financial models
# Will not be used by quant workflow
# TODO the comment above is not true cause this module is imported in the crocubot.train module.
# TODO move this iotools somewhere in the benchmark namespace

import os
import numpy as np
import tensorflow as tf

from alphai_data_sources.generator import BatchGenerator
from alphai_crocubot_oracle.data.classifier import classify_labels

FLAGS = tf.app.flags.FLAGS
batch_generator = BatchGenerator()


def reset_mnist():
    batch_generator.reset_mnist()


def load_batch(batch_options, data_source, bin_edges=None):

    features, labels = batch_generator.get_batch(batch_options, data_source)

    if bin_edges is not None:
        labels = classify_labels(bin_edges, labels)

    # Kernel dimension, now that crocubot is 4D
    features = np.expand_dims(features, axis=1)
    labels = np.expand_dims(labels, axis=1)

    return features, labels


def load_file_name(series_name, topology):
    """ File used for storing the network parameters.

    :param str series_name: Identify the data on which the network was trained: MNIST, low_noise, randomwalk, etc
    :param Topology topology: Info on network shape
    :return:
    """

    depth_string = str(topology.n_layers)
    breadth_string = str(topology.n_timesteps)
    series_string = str(topology.n_series)

    bitstring = str(FLAGS.TF_TYPE)
    path = FLAGS.model_save_path

    file_name = "{}model_{}_{}_{}x{}.ckpt".format(bitstring[-2:], series_name, series_string, depth_string,
                                                  breadth_string)
    return os.path.join(path, file_name)
