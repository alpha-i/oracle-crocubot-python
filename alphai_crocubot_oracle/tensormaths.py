# TODO this tensormath is used to load the activation function inside the Topology.
# TODO create an activation function provider service

import tensorflow as tf
import numpy as np
from alphai_time_series.transform import gaussianise

LOG_TWO_PI = np.log(2 * np.pi)
MIN_LOG_LIKELIHOOD = -10
DEFAULT_TF_TYPE = tf.float32
DEFAULT_D_TYPE = 'float32'  # FIXME these cannot be set here. We need to move this somewhere else.


def selu(x):
    """
    selu activation function
    see: https://arxiv.org/pdf/1706.02515.pdf
    :param x: A Tensor.
    :return: Has the same type as x.
    """
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))


def inv_selu(x):
    """ Inverse of the selu activation function, such that inv_selu(selu(x)) = x.

     For x>0, f(x)= scale * x so f^-1(x) = x / scale
     For x<0, f(x) = scale * alpha * (exp^x - 1) so f^-1(x) = log(x/alpha/scale + 1)
     N.B. Becomes pathological for x < -alpha*scale
    :param x: A Tensor.
    :return: Has the same type as x.
    """

    epsilon = 1e-3
    close_to_unity = 1 - epsilon
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946

    xmin = -close_to_unity * alpha * scale
    xmax = 1e100  # For some reason clip_value_max is mandatory

    x = tf.clip_by_value(x, clip_value_min=xmin, clip_value_max=xmax)

    return tf.where(x >= 0.0, x / scale, tf.log(x / alpha / scale + 1.))


def kelu(x):
    """
    experimental activation function
    :param x: A Tensor.
    :return: Has the same type as x.
    """
    k = 3.0
    return tf.where(x >= 0.0, k * x, x / k)


def inv_kelu(x):
    """
    inverse of kelu function
    :param x: A Tensor.
    :return: Has the same type as x.
    """
    return -kelu(-x)


def centred_gaussian(shape, sigma=1., seed=None):
    """
    Useful for generating Gaussian noise.
    :param shape: Shape of the noise Tensor to be generated.
    :param sigma: The standard deviation of the Gaussian.
    :param seed: Random number generator seed.
    :return: Gaussian distributed random variates.
    """
    return tf.random_normal(shape=shape, mean=0., stddev=sigma, seed=seed, dtype=DEFAULT_TF_TYPE)


def perfect_centred_gaussian(shape, sigma=1.):
    """
    Useful for generating de-noised Gaussian noise where the noise on the moments is zero.
    Unfortunately tensorflow currently lacks erfinv so this must be done in numpy.
    :param shape: Shape of the noise Tensor to be generated.
    :param sigma: The standard deviation of the Gaussian.
    :return: Perfectly centered Gaussian random variates.
    """
    rand_noise = np.random.normal(loc=0., scale=sigma, size=shape)
    perfect_noise = gaussianise(rand_noise, target_mean=0., target_sigma=sigma, do_uniform_sampling=True)

    return perfect_noise.astype(DEFAULT_D_TYPE)


def log_gaussian(x, mu, sigma):
    """
    Log probability density at x given mean and standard deviation.
    :param x: The position at which the probability is to be calculated.
    :param mu: The mean of the distribution.
    :param sigma: The standard deviation of the distribution.
    :return: The log-probability value.
    """
    return -0.5 * LOG_TWO_PI - tf.log(sigma) - (x - mu) ** 2 / (2. * sigma ** 2)


def log_gaussian_logsigma(x, mu, logsigma):
    """
    Same as log_gaussian but here takes log of sigma as input
    :param x: The position at which the probability is to be calculated.
    :param mu: The mean of the distribution.
    :param logsigma: The logarithm of the standard deviation of the distribution.
    :return: The log-probability value.
    """

    return -0.5 * LOG_TWO_PI - logsigma - (x - mu) ** 2 / (2. * tf.exp(2 * logsigma))


def unit_gaussian(x):
    """
    Compute the probability of the Unit Gaussian variate.
    :param x: The position at which the probability is requested.
    :return:  The probability at x.
    """
    return tf.cast(tf.contrib.distributions.Normal(0., 1.).prob(x), DEFAULT_TF_TYPE)


def sinh_shift(x, c):
    """
    Activation function sinh_shift. See http://www.leemon.com/papers/2005bsi.pdf
    :param x: A Tensor.
    :param c: The offset or shift parameter.
    :return: Has the same type as x.
    """
    pos_t = 0.5 + c * tf.exp(-x) - tf.exp(-2 * x) / 2
    neg_t = -0.5 + c * tf.exp(x) + tf.exp(2 * x) / 2

    pos_f = x + tf.log(pos_t + tf.sqrt(tf.exp(-2 * x) + tf.square(pos_t)))
    neg_f = x - tf.log(-neg_t + tf.sqrt(tf.exp(2 * x) + tf.square(neg_t)))

    return tf.where(x > 0.0, pos_f, neg_f)
