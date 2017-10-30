import tensorflow as tf
import numpy as np
from scipy.stats import norm

from alphai_crocubot_oracle.bayesian_cost import BayesianCost
from alphai_crocubot_oracle.topology import Topology
from alphai_crocubot_oracle.crocubot.model import CrocuBotModel
from tests.helpers import get_default_flags


class TestBayesianCost(tf.test.TestCase):

    def test_calc_log_q_prior(self):

        parameters = [
            (1., 0., 1., -1.4813652),
            (2., -1., -1., -45.614418)
        ]
        parameters = np.array(parameters, dtype=np.float32)
        with self.test_session():
            for test in parameters:
                theta, mu, rho, expected_result = test
                actual_result = BayesianCost.calculate_log_q_prior(theta, mu, rho).eval()
                self.assertAlmostEqual(
                    actual_result,
                    expected_result,
                    places=4
                )

    def test_init(self):

        flags = get_default_flags()

        layer_number = [
            {"activation_func": "relu", "trainable": False, "height": 20, "width": 10, "cell_height": 1},
            {"activation_func": "relu", "trainable": False, "height": 20, "width": 10, "cell_height": 1},
            {"activation_func": "linear", "trainable": False, "height": 20, "width": 10, "cell_height": 1}
        ]
        topology = Topology(layer_number)

        self.model = CrocuBotModel(topology, flags)

        # case1 no error thrown
        use_double_gaussian_weights_prior = True
        slab_std_dvn = 1.2
        spike_std_dvn = 0.05
        spike_slab_weighting = 0.5
        BayesianCost(self.model, use_double_gaussian_weights_prior, slab_std_dvn, spike_std_dvn, spike_slab_weighting)

        # case2 slab_std_dvn < 0
        slab_std_dvn = -1.
        self.assertRaises(ValueError, BayesianCost, model=self.model,
                          use_double_gaussian_weights_prior=use_double_gaussian_weights_prior,
                          slab_std_dvn=slab_std_dvn, spike_std_dvn=spike_std_dvn,
                          spike_slab_weighting=spike_slab_weighting)

        # case3 slab_std_dvn > 100
        slab_std_dvn = 101.
        self.assertRaises(ValueError, BayesianCost, model=self.model,
                          use_double_gaussian_weights_prior=use_double_gaussian_weights_prior,
                          slab_std_dvn=slab_std_dvn, spike_std_dvn=spike_std_dvn,
                          spike_slab_weighting=spike_slab_weighting)

        # case4 spike_std_dvn < 0
        spike_std_dvn = -1.
        self.assertRaises(ValueError, BayesianCost, model=self.model,
                          use_double_gaussian_weights_prior=use_double_gaussian_weights_prior,
                          slab_std_dvn=slab_std_dvn, spike_std_dvn=spike_std_dvn,
                          spike_slab_weighting=spike_slab_weighting)

        # case5 spike_std_dvn > 100
        spike_std_dvn = 101.
        self.assertRaises(ValueError, BayesianCost, model=self.model,
                          use_double_gaussian_weights_prior=use_double_gaussian_weights_prior,
                          slab_std_dvn=slab_std_dvn, spike_std_dvn=spike_std_dvn,
                          spike_slab_weighting=spike_slab_weighting)

        # case6 spike_std_dvn > slab_std_dvn
        spike_std_dvn = 5.
        slab_std_dvn = 1.
        self.assertRaises(ValueError, BayesianCost, model=self.model,
                          use_double_gaussian_weights_prior=use_double_gaussian_weights_prior,
                          slab_std_dvn=slab_std_dvn, spike_std_dvn=spike_std_dvn,
                          spike_slab_weighting=spike_slab_weighting)

        # case7 spike_slab_weighting < 0
        spike_slab_weighting = -1.
        self.assertRaises(ValueError, BayesianCost, model=self.model,
                          use_double_gaussian_weights_prior=use_double_gaussian_weights_prior,
                          slab_std_dvn=slab_std_dvn, spike_std_dvn=spike_std_dvn,
                          spike_slab_weighting=spike_slab_weighting)

        # case8 spike_slab_weighting > 1.
        spike_slab_weighting = 2.
        self.assertRaises(ValueError, BayesianCost, model=self.model,
                          use_double_gaussian_weights_prior=use_double_gaussian_weights_prior,
                          slab_std_dvn=slab_std_dvn, spike_std_dvn=spike_std_dvn,
                          spike_slab_weighting=spike_slab_weighting)

    def test_calculate_log_weight_prior(self):

        self.test_init()

        with self.test_session():
            layer = 0
            layer_name = str(layer)
            alpha_value = 0.2
            log_alpha_value = np.log(alpha_value).astype(np.float32)
            init_log_alpha = tf.constant_initializer(log_alpha_value)
            with tf.variable_scope(layer_name):
                log_alpha = tf.get_variable('log_alpha', shape=(), initializer=init_log_alpha)

            log_alpha.initializer.run()
            self.assertEqual(log_alpha.eval(), log_alpha_value)
            log_alpha_retrieved = self.model.get_variable(0, self.model.VAR_LOG_ALPHA)
            # TODO this a now a test get_layer_variable()
            self.assertEqual(log_alpha_retrieved.eval(), log_alpha_value)

            # case 1 slab prior
            use_double_gaussian_weights_prior = True
            slab_std_dvn = 1.
            spike_std_dvn = 0.05
            spike_slab_weighting = 1.
            bayes_cost = BayesianCost(self.model, use_double_gaussian_weights_prior, slab_std_dvn, spike_std_dvn,
                                      spike_slab_weighting)
            weights = np.random.normal(size=5)
            weights = weights.astype(np.float32)
            log_prior_value_computed = bayes_cost.calculate_log_weight_prior(weights, layer)
            log_prior_value_expected = np.sum(norm.logpdf(weights))
            self.assertAlmostEqual(log_prior_value_computed.eval(), log_prior_value_expected, places=5)

            # case 2 spike prior
            slab_std_dvn = 2.  # note that we have condition that slab_std_dvn >= spike_std_dvn
            spike_std_dvn = 1.
            spike_slab_weighting = 0.
            bayes_cost = BayesianCost(self.model, use_double_gaussian_weights_prior, slab_std_dvn, spike_std_dvn,
                                      spike_slab_weighting)
            log_prior_value_computed = bayes_cost.calculate_log_weight_prior(weights, layer)
            self.assertAlmostEqual(log_prior_value_computed.eval(), log_prior_value_expected, places=5)

            # case 3 50/50 spike/slab
            slab_std_dvn = 2.
            spike_std_dvn = 1.
            spike_slab_weighting = 0.5
            bayes_cost = BayesianCost(self.model, use_double_gaussian_weights_prior, slab_std_dvn, spike_std_dvn,
                                      spike_slab_weighting)
            log_prior_value_computed = bayes_cost.calculate_log_weight_prior(weights, layer)
            log_prior_value_expected = np.sum(np.log(spike_slab_weighting * norm.pdf(weights / slab_std_dvn) /
                                                     slab_std_dvn + (1. - spike_slab_weighting) *
                                                     norm.pdf(weights / spike_std_dvn) / spike_std_dvn))
            self.assertAlmostEqual(log_prior_value_computed.eval(), log_prior_value_expected, places=5)

            # case 4 no double Gaussian prior
            use_double_gaussian_weights_prior = False
            bayes_cost = BayesianCost(self.model, use_double_gaussian_weights_prior, slab_std_dvn, spike_std_dvn,
                                      spike_slab_weighting)
            log_prior_value_computed = bayes_cost.calculate_log_weight_prior(weights, layer)
            log_prior_value_expected = np.sum(norm.logpdf(weights, scale=alpha_value))
            self.assertAlmostEqual(log_prior_value_computed.eval(), log_prior_value_expected, places=5)

    def test_calculate_log_bias_prior(self):
        # FIXME the only difference between this and the previous one is the function
        # FIXME calculate_log_weight_prior() changes to calculate_log_weight_prior().
        # FIXME Otherwise, they are identical.

        self.test_init()

        with self.test_session():
            layer = 0
            layer_name = str(layer)
            alpha_value = 0.2
            log_alpha_value = np.log(alpha_value).astype(np.float32)
            init_log_alpha = tf.constant_initializer(log_alpha_value)
            with tf.variable_scope(layer_name):
                log_alpha = tf.get_variable('log_alpha', shape=(), initializer=init_log_alpha)

            log_alpha.initializer.run()
            self.assertEqual(log_alpha.eval(), log_alpha_value)
            log_alpha_retrieved = self.model.get_variable(0, self.model.VAR_LOG_ALPHA)
            # TODO this a now a test get_layer_variable()
            self.assertEqual(log_alpha_retrieved.eval(), log_alpha_value)

            # case 1 slab prior
            use_double_gaussian_weights_prior = True
            slab_std_dvn = 1.
            spike_std_dvn = 0.05
            spike_slab_weighting = 1.
            bayes_cost = BayesianCost(self.model, use_double_gaussian_weights_prior, slab_std_dvn, spike_std_dvn,
                                      spike_slab_weighting)
            weights = np.random.normal(size=5)
            weights = weights.astype(np.float32)
            log_prior_value_computed = bayes_cost.calculate_log_bias_prior(weights, layer)
            log_prior_value_expected = np.sum(norm.logpdf(weights))
            self.assertAlmostEqual(log_prior_value_computed.eval(), log_prior_value_expected, places=5)

            # case 2 spike prior
            slab_std_dvn = 2.  # note that we have condition that slab_std_dvn >= spike_std_dvn
            spike_std_dvn = 1.
            spike_slab_weighting = 0.
            bayes_cost = BayesianCost(self.model, use_double_gaussian_weights_prior, slab_std_dvn, spike_std_dvn,
                                      spike_slab_weighting)
            log_prior_value_computed = bayes_cost.calculate_log_weight_prior(weights, layer)
            self.assertAlmostEqual(log_prior_value_computed.eval(), log_prior_value_expected, places=5)

            # case 3 50/50 spike/slab
            slab_std_dvn = 2.
            spike_std_dvn = 1.
            spike_slab_weighting = 0.5
            bayes_cost = BayesianCost(self.model, use_double_gaussian_weights_prior, slab_std_dvn, spike_std_dvn,
                                      spike_slab_weighting)
            log_prior_value_computed = bayes_cost.calculate_log_bias_prior(weights, layer)
            log_prior_value_expected = np.sum(np.log(spike_slab_weighting * norm.pdf(weights / slab_std_dvn) /
                                                     slab_std_dvn + (1. - spike_slab_weighting) *
                                                     norm.pdf(weights / spike_std_dvn) / spike_std_dvn))
            self.assertAlmostEqual(log_prior_value_computed.eval(), log_prior_value_expected, places=5)

            # case 4 no double Gaussian prior
            use_double_gaussian_weights_prior = False
            bayes_cost = BayesianCost(self.model, use_double_gaussian_weights_prior, slab_std_dvn, spike_std_dvn,
                                      spike_slab_weighting)
            log_prior_value_computed = bayes_cost.calculate_log_bias_prior(weights, layer)
            log_prior_value_expected = np.sum(norm.logpdf(weights, scale=alpha_value))
            self.assertAlmostEqual(log_prior_value_computed.eval(), log_prior_value_expected, places=5)

    def test_calculate_log_hyperprior(self):

        self.test_init()

        with self.test_session():
            layer = 0
            layer_name = str(layer)
            alpha_value = 0.2
            log_alpha_value = np.log(alpha_value).astype(np.float32)
            init_log_alpha = tf.constant_initializer(log_alpha_value)
            with tf.variable_scope(layer_name):
                log_alpha = tf.get_variable(self.model.VAR_LOG_ALPHA, shape=(), initializer=init_log_alpha)

            log_alpha.initializer.run()
            self.assertEqual(log_alpha.eval(), log_alpha_value)
            log_alpha_retrieved = self.model.get_variable(0, self.model.VAR_LOG_ALPHA)
            # TODO this a now a test get_variable()
            self.assertEqual(log_alpha_retrieved.eval(), log_alpha_value)

            # case 1 test the hyper prior
            use_double_gaussian_weights_prior = True
            slab_std_dvn = 1.
            spike_std_dvn = 0.05
            spike_slab_weighting = 1.
            bayes_cost = BayesianCost(self.model, use_double_gaussian_weights_prior, slab_std_dvn, spike_std_dvn,
                                      spike_slab_weighting)
            log_prior_value_computed = bayes_cost.calculate_log_hyperprior(layer)
            log_prior_value_expected = - log_alpha_value
            self.assertAlmostEqual(log_prior_value_computed.eval(), log_prior_value_expected, places=5)
