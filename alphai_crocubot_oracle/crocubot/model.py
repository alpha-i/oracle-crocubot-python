"""
This modules contains two classes

1 CrocuBotModel
    class which wraps all the methods needed to initialize the graph and retrieve its variables
        The graph in which the class operates is the default tf graph
2 Estimator
  This class is an service class which permits to visit the graph of a given estimator and retrieve different passes
   algrithm


"""

import logging

import numpy as np
import tensorflow as tf

import alphai_crocubot_oracle.tensormaths as tm


class CrocuBotModel:

    VAR_WEIGHT_RHO = 'rho_w'
    VAR_WEIGHT_MU = 'mu_w'
    VAR_WEIGHT_NOISE = 'weight_noise'

    VAR_BIAS_RHO = 'rho_b'
    VAR_BIAS_MU = 'mu_b'
    VAR_BIAS_NOISE = 'bias_noise'

    VAR_LOG_ALPHA = 'log_alpha'

    def __init__(self, topology, flags):

        self._topology = topology
        self._graph = tf.get_default_graph()
        self._flags = flags

    @property
    def graph(self):
        return self._graph

    @property
    def topology(self):
        return self._topology

    @property
    def number_of_layers(self):
        return self._topology.n_layers

    @property
    def layer_variables_list(self):
        return [
            self.VAR_BIAS_RHO,
            self.VAR_BIAS_MU,
            self.VAR_BIAS_NOISE,
            self.VAR_WEIGHT_RHO,
            self.VAR_WEIGHT_MU,
            self.VAR_WEIGHT_NOISE,
            self.VAR_LOG_ALPHA
        ]

    def build_layers_variables(self):

        weight_uncertainty = self._flags.INITIAL_WEIGHT_UNCERTAINTY
        bias_uncertainty = self._flags.INITIAL_BIAS_UNCERTAINTY
        weight_displacement = self._flags.INITIAL_WEIGHT_DISPLACEMENT
        bias_displacement = self._flags.INITIAL_BIAS_DISPLACEMENT

        initial_rho_weights = tf.contrib.distributions.softplus_inverse(weight_uncertainty)
        initial_rho_bias = tf.contrib.distributions.softplus_inverse(bias_uncertainty)
        initial_alpha = self._flags.INITIAL_ALPHA

        for layer_number in range(self._topology.n_layers):

            w_shape = self._topology.get_weight_shape(layer_number)
            b_shape = self._topology.get_bias_shape(layer_number)

            self._create_variable_for_layer(
                layer_number,
                self.VAR_WEIGHT_MU,
                tm.centred_gaussian(w_shape, weight_displacement)
            )

            self._create_variable_for_layer(
                layer_number,
                self.VAR_WEIGHT_RHO,
                initial_rho_weights + tm.centred_gaussian(w_shape, np.abs(initial_rho_weights) / 10)
            )

            self._create_variable_for_layer(
                layer_number,
                self.VAR_BIAS_MU,
                tm.centred_gaussian(b_shape, bias_displacement)
            )

            self._create_variable_for_layer(
                layer_number,
                self.VAR_BIAS_RHO,
                initial_rho_bias + tm.centred_gaussian(b_shape, np.abs(initial_rho_bias) / 10)
            )

            self._create_variable_for_layer(
                layer_number,
                self.VAR_LOG_ALPHA,
                np.log(initial_alpha).astype(self._flags.d_type),
                False
            )  # Hyperprior on the distribution of the weights

            self._create_noise(layer_number, self.VAR_WEIGHT_NOISE, w_shape)
            self._create_noise(layer_number, self.VAR_BIAS_NOISE, b_shape)

    def _create_variable_for_layer(self, layer_number, variable_name, initializer, is_trainable=True):

        assert isinstance(layer_number, int)
        scope_name = str(layer_number)
        with tf.variable_scope(scope_name):  # TODO check if this is the correct
            tf.get_variable(variable_name, initializer=initializer, trainable=is_trainable, dtype=tm.DEFAULT_TF_TYPE)

    def _create_noise(self, layer_number, variable_name, shape):

        if self._flags.USE_PERFECT_NOISE:
            noise_vector = tm.perfect_centred_gaussian(shape)
        else:
            noise_vector = tm.centred_gaussian(shape)

        self._create_variable_for_layer(layer_number, variable_name, noise_vector, False)

    def get_variable(self, layer_number, variable_name, reuse=True):

        scope_name = str(layer_number)
        with tf.variable_scope(scope_name, reuse=reuse):
            v = tf.get_variable(variable_name, dtype=tm.DEFAULT_TF_TYPE)

        return v

    def get_weight_noise(self, layer_number, iteration):
        noise = self.get_variable(layer_number, self.VAR_WEIGHT_NOISE)
        return tm.roll_noise(
            noise,
            iteration
        )

    def get_bias_noise(self, layer_number, iteration):
        noise = self.get_variable(layer_number, self.VAR_BIAS_NOISE)
        return tm.roll_noise(
            noise,
            iteration
        )

    def compute_weights(self, layer_number, iteration=0):

        mean = self.get_variable(layer_number, self.VAR_WEIGHT_MU)
        rho = self.get_variable(layer_number, self.VAR_WEIGHT_RHO)
        noise = self.get_weight_noise(layer_number, iteration)

        return mean + tf.nn.softplus(rho) * noise
        #  return mean + tf.exp(rho) * noise

    def compute_biases(self, layer_number, iteration):
        """Bias is Gaussian distributed"""
        mean = self.get_variable(layer_number, self.VAR_BIAS_MU)
        rho = self.get_variable(layer_number, self.VAR_BIAS_RHO)
        noise = self.get_bias_noise(layer_number, iteration)

        return mean + tf.nn.softplus(rho) * noise
        #  return mean + tf.exp(rho) * noise


class Estimator:

    def __init__(self, crocubot_model, flags):
        """

        :param CrocuBotModel crocubot_model:
        :param flags:
        :return:
        """
        self._model = crocubot_model
        self._flags = flags

    def average_multiple_passes(self, data, number_of_passes):
        """
        Multiple passes allow us to estimate the posterior distribution.

        :param data: Mini-batch to be fed into the network
        :param number_of_passes: How many random realisations of the weights should be sampled
        :return: Means and variances of the posterior. NB this is not the covariance - see network_covariance.py
        """

        collated_outputs = self.collate_multiple_passes(data, number_of_passes)
        mean, variance = tf.nn.moments(collated_outputs, axes=[0])

        if number_of_passes == 1:
            logging.warning("Using default variance")
            variance = self._flags.DEFAULT_FORECAST_VARIANCE

        return mean, variance

    def collate_multiple_passes(self, x, number_of_passes=50):
        """
        Collate outputs from many realisations of weights from a bayesian network.

        :param tensor x:
        :param int number_of_passes:
        :return 4D tensor with dimensions [n_passes, batch_size, n_label_timesteps, n_categories]:
        """

        outputs = []
        for iteration in range(number_of_passes):
            result = self.forward_pass(x, iteration)
            outputs.append(result)

        stacked_output = tf.stack(outputs, axis=0)

        # Make sure we softmax across the 'bin' dimension, but not across all series!
        stacked_output = tf.nn.softmax(stacked_output, dim=-1)

        return stacked_output

    def forward_pass(self, signal, iteration=0):
        """
        Takes input data and returns predictions

        :param tensor signal: signal[i,j] holds input j from sample i, so data.shape = [batch_size, n_inputs]
        or if classification then   [batch_size, n_series, n_classes]
        :param int iteration: number of iteration
        :return:
        """

        for layer_number in range(self._model.topology.n_layers):
            weights = self._model.compute_weights(layer_number, iteration)
            biases = self._model.compute_biases(layer_number, iteration)

            signal = tf.tensordot(signal, weights, axes=2) + biases
            activation_function = self._model.topology.get_activation_function(layer_number)
            signal = activation_function(signal)

        return signal
