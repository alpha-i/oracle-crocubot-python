"""
This modules contains the implementation of the BayesianCost which is
the cost function to control the learning process.

It is used during the train of the model, implemented in the module alphai_crocubot_oracle.train:train

"""
import tensorflow as tf

import alphai_crocubot_oracle.tensormaths as tm


class BayesianCost(object):
    def __init__(self, model, double_gaussian_weights_prior, wide_prior_std, narrow_prior_std,
                 spike_slab_weighting):
        """

        :param alphai_crocubot_oracle.crocubot.model.CrocuBotModel model:
        :param Topology topology:
        :param double_gaussian_weights_prior:
        :param wide_prior_std:
        :param narrow_prior_std:
        :param spike_slab_weighting:
        """
        self._model = model
        self.topology = model.topology
        self._double_gaussian_weights_prior = double_gaussian_weights_prior
        self._wide_prior_std = wide_prior_std
        self._narrow_prior_std = narrow_prior_std
        self._spike_slab_weighting = spike_slab_weighting

        assert 0.0 < self._spike_slab_weighting < 1.0, " spike_slab_weighting must be between 0 and 1"

    def get_bayesian_cost(self, prediction, target):
        log_pw, log_qw = self.calculate_priors()
        log_likelihood = self.calculate_likelihood(prediction, target)

        return log_qw - log_pw - log_likelihood

    def calculate_priors(self):
        log_pw = 0.
        log_qw = 0.

        for layer in range(self.topology.n_layers):
            mu_w = self._model.get_variable(layer, self._model.VAR_WEIGHT_MU)
            rho_w = self._model.get_variable(layer, self._model.VAR_WEIGHT_RHO)
            mu_b = self._model.get_variable(layer, self._model.VAR_BIAS_MU)
            rho_b = self._model.get_variable(layer, self._model.VAR_BIAS_RHO)

            # Only want to consider independent weights, not the full set, so do_tile_weights=False
            weights = self._model.compute_weights(layer, iteration=0)
            biases = self._model.compute_biases(layer, iteration=0)

            log_pw += self.calc_log_weight_prior(weights, layer)  # not needed if we're using many passes
            log_pw += self.calc_log_bias_prior(biases, layer)
            log_pw += self.calc_log_hyperprior(layer)

            log_qw += self.calc_log_q_prior(weights, mu_w, rho_w)
            log_qw += self.calc_log_q_prior(biases, mu_b, rho_b)

        return log_pw, log_qw

    def calc_log_weight_prior(self, weights, layer):  # TODO can we make these two into a single function?
        """
        See Equation 7 in https://arxiv.org/pdf/1505.05424.pdf
        :param weights:
        :param layer:
        :return:
        """
        if self._double_gaussian_weights_prior:

            pwide = tm.unit_gaussian(weights / self._wide_prior_std) / self._wide_prior_std
            pnarrow = tm.unit_gaussian(weights / self._narrow_prior_std) / self._narrow_prior_std

            log_pw = tf.log(self._spike_slab_weighting * pwide + (1 - self._spike_slab_weighting) * pnarrow)
        else:
            log_alpha = self._model.get_variable(layer, self._model.VAR_LOG_ALPHA)
            log_pw = tm.log_gaussian_logsigma(weights, 0., log_alpha)

        return tf.reduce_sum(log_pw)

    def calc_log_bias_prior(self, biases, layer):  # TODO can we make these two into a single function?
        """
        See Equation 7 in https://arxiv.org/pdf/1505.05424.pdf
        :param biases:
        :param layer:
        :return:
        """
        if self._double_gaussian_weights_prior:

            pwide = tf.contrib.distributions.Normal(0., 1.).prob(biases / self._wide_prior_std) / self._wide_prior_std
            pnarrow = tf.contrib.distributions.Normal(0., 1.).prob(biases / self._narrow_prior_std) / self._narrow_prior_std

            log_pw = tf.log(self._spike_slab_weighting * pwide + (1 - self._spike_slab_weighting) * pnarrow)
        else:
            log_alpha = self._model.get_variable(layer, self._model.VAR_LOG_ALPHA)
            log_pw = tm.log_gaussian_logsigma(biases, 0., log_alpha)

        return tf.reduce_sum(log_pw)

    def calc_log_hyperprior(self, layer):
        return - self._model.get_variable(layer, self._model.VAR_LOG_ALPHA)  # p(alpha) = 1 / alpha so log(p(alpha)) = - log(alpha)

    @staticmethod
    def calc_log_q_prior(theta, mu, rho):

        # sigma = tf.nn.softplus(rho) #
        sigma = tf.exp(rho)
        log_qw = tm.log_gaussian(theta, mu, sigma)

        return tf.reduce_sum(log_qw)

    def calculate_likelihood(self, truth, forecast):
        tm.MIN_LOG_LIKELIHOOD = -10  # Avoid numerical issues
        true_indices = tf.argmax(truth, axis=2)  # Dimensions [batch_size, N_LABEL_TIMESTEPS, N_LABEL_CLASSES]
        p_forecast = tf.gather(forecast, true_indices)
        log_likelihood = tf.maximum(tf.log(p_forecast), tm.MIN_LOG_LIKELIHOOD)

        return tf.reduce_sum(log_likelihood)
