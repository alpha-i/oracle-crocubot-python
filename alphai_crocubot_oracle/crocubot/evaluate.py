# This module is used to make predictions
# Only used by oracle.py

import logging

import numpy as np
import tensorflow as tf

# FIXME once time_series is updated, uncomment the below and delete the copy in this file
# from alphai_time_series.calculator import make_diagonal_covariance_matrices

import alphai_crocubot_oracle.classifier as cl
from alphai_crocubot_oracle.crocubot.model import CrocuBotModel, Estimator

FLAGS = tf.app.flags.FLAGS


def eval_neural_net(data, topology, save_file):
    """ Multiple passes allow us to estimate the posterior distribution.

    :param data:  Mini-batch to be fed into the network
    :param topology: Specifies layout of network, also used to identify save file
    :return: 3D array with dimensions [n_passes, n_samples, n_labels]
    """

    logging.info("Evaluating with shape", data.shape)

    model = CrocuBotModel(topology, FLAGS)
    try:
        model.build_layers_variables()
    except:
        logging.info('Variables already initialised')

    saver = tf.train.Saver()
    estimator = Estimator(model, FLAGS)
    y = estimator.collate_multiple_passes(data, FLAGS.n_eval_passes)

    with tf.Session() as sess:
        logging.info("Attempting to recover trained network:", save_file)
        saver.restore(sess, save_file)

        return y.eval()


def forecast_means_and_variance(outputs, bin_distribution):
    """ Each forecast comprises a mean and variance. NB not the covariance matrix
    Oracle will perform this outside, but this function is useful for testing purposes

    :param nparray outputs: Raw output from the network, a 4D array of shape [n_passes, n_samples, n_series, classes]
    :param bin_distribution: Characterises the binning used to perform the classification task
    :return: Means and variances of the posterior.
    """

    assert outputs.shape[0] == FLAGS.n_eval_passes, 'unexpected output shape'
    n_samples = outputs.shape[1]
    n_series = outputs.shape[2]

    mean = np.zeros(shape=(n_samples, n_series))
    variance = np.zeros(shape=(n_samples, n_series))

    for i in range(n_samples):
        for j in range(n_series):
            bin_passes = outputs[:, i, j, :]
            temp_mean, temp_variance = cl.declassify_labels(bin_distribution, bin_passes)
            mean[i, j] = temp_mean
            variance[i, j] = temp_variance

    if n_series > 1:
        variance = make_diagonal_covariance_matrices(variance)

    return mean, variance


# FIXME delete me once available in alphai_time_series
def make_diagonal_covariance_matrices(variances):
    """ Takes array of variances and makes diagonal covariance matrices

    :param variances: [i, j] holds variance of forecast of sample i and series j
    :return: Array of covariance matrices [n_samples, n_series, n_series]
    """

    if variances.ndim != 2:
        raise ValueError('Dimensionality of the variances matrix {} should be 2'.format(variances.ndim))

    n_samples = variances.shape[0]
    n_series = variances.shape[1]

    covariance_matrices = np.zeros((n_samples, n_series, n_series))

    for i in range(n_samples):
        diagonal_terms = variances[i, :]
        covariance_matrices[i, :, :] = np.diag(diagonal_terms)

    return covariance_matrices
