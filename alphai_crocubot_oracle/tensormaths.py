# Library of mathematical functions for use with tensorflow
import tensorflow as tf
import numpy as np

from alphai_time_series.transform import gaussianise

LOG_TWO_PI = np.log(2 * np.pi)

# TODO
# 1 set variables set in the config file and not as global constants
# 2 make doctrings for all functions

MIN_LOG_LIKELIHOOD = -10
DEFAULT_TF_TYPE = tf.float32
DEFAULT_D_TYPE = 'float32'


def selu(x):
    """
    selu activation function 
    see: https://arxiv.org/pdf/1706.02515.pdf
    :param x: 
    :return:
    """
    alpha = 1.6732632423543772848170429916717
    scale = 1.0507009873554804934193349852946
    return scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))


def inv_selu(x):
    """ Inverse of the selu activation function, such that inv_selu(selu(x)) = x.

     For x>0, f(x)= scale * x so f^-1(x) = x / scale
     For x<0, f(x) = scale * alpha * (exp^x - 1) so f^-1(x) = log(x/alpha/scale + 1)
     N.B. Becomes pathological for x < -alpha*scale
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
    :param x: 
    :return: 
    """
    k = 3.0
    return tf.where(x >= 0.0, k * x, x / k)


def inv_kelu(x):
    """
    inverse of kelu function 
    :param x: 
    :return: 
    """
    return -kelu(-x)


def centred_gaussian(shape, sigma=1., seed=None):
    """Useful for generating Gaussian noise"""
    return tf.random_normal(shape=shape, mean=0., stddev=sigma, seed=seed, dtype=DEFAULT_TF_TYPE)


def perfect_centred_gaussian(shape, sigma=1.):
    """Useful for generating de-noised Gaussian noise where the noise on the moments is zero
    Unfortunately tensorflow currently lacks erfinv so this must be done in numpy"""

#    rand_noise = tf.random_normal(shape=shape, mean=0., stddev=sigma, seed=seed)
    rand_noise = np.random.normal(loc=0., scale=sigma, size=shape)
    perfect_noise = gaussianise(rand_noise, target_mean=0., target_sigma=sigma, do_uniform_sampling=True)

    return perfect_noise.astype(DEFAULT_D_TYPE)


def log_gaussian(x, mu, sigma):
    """Log probability density at x given mean and standard deviation"""
    return -0.5 * LOG_TWO_PI - tf.log(sigma) - (x - mu) ** 2 / (2. * sigma ** 2)


def log_gaussian_logsigma(x, mu, logsigma):
    """Same as log_gaussian but here takes log of sigma as input"""

    return -0.5 * LOG_TWO_PI - logsigma - (x - mu) ** 2 / (2. * tf.exp(2 * logsigma))


def unit_gaussian(x):
    return tf.cast(tf.contrib.distributions.Normal(0., 1.).prob(x), DEFAULT_TF_TYPE)


def sinh_shift(x, c):
    """ See http://www.leemon.com/papers/2005bsi.pdf

    :param x:
    :param c:
    :return:
    """
    pos_t = 0.5 + c*tf.exp(-x) - tf.exp(-2 * x) / 2
    neg_t = -0.5 + c*tf.exp(x) + tf.exp(2 * x) / 2

    pos_f = x + tf.log(pos_t + tf.sqrt(tf.exp(-2 * x) + tf.square(pos_t)))
    neg_f = x - tf.log(-neg_t + tf.sqrt(tf.exp(2 * x) + tf.square(neg_t)))

    return tf.where(x > 0.0, pos_f, neg_f)


def roll_noise(noise, iteration):
    """Wraps noise iteration in tensorflow framework"""

    return tf.py_func(roll_noise_np, [noise, iteration], DEFAULT_TF_TYPE)


def roll_noise_np(noise, iteration):
    """

    :param noise: numpy array 
    :param iteration: type integer 
    :return:
    """

    shift_factor = 1
    shift = iteration * shift_factor
    original_shape = noise.shape
    noise = noise.flatten()
    noise = np.roll(noise, shift=shift)

    return np.reshape(noise, original_shape)
