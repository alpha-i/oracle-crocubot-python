import tensorflow as tf
import numpy as np

from alphai_crocubot_oracle.crocubot.model import CrocuBotModel, Estimator

from alphai_crocubot_oracle.topology import (
    Topology,
    DEFAULT_HIDDEN_LAYERS,
    DEFAULT_DEPTH,
    DEFAULT_N_SERIES,
    DEFAULT_TIMESTEPS,
    DEFAULT_BINS,
    DEFAULT_N_FORECASTS
)
from tests.helpers import get_default_flags

DEFAULT_BATCH_SIZE = 100


class TestCrocuBotModel(tf.test.TestCase):

    def test_create_model(self):

        flags = get_default_flags()

        topology = Topology()

        model = CrocuBotModel(topology, flags, is_training=True)
        n_connections_between_layers = DEFAULT_HIDDEN_LAYERS + 1
        self.assertEqual(model.number_of_layers, n_connections_between_layers)
        self.assertEqual(model.topology, topology)

        with self.test_session() as session:
            for layer_number in range(model.number_of_layers):
                for variable_name in model.layer_variables_list:
                    self.assertRaises(
                        ValueError,
                        model.get_variable,
                        layer_number, variable_name
                    )

        with self.test_session() as session:

            model.build_layers_variables()
            session.run(tf.global_variables_initializer())

            self.assertEqual(len(session.run(tf.report_uninitialized_variables())), 0)

            for layer_number in range(model.number_of_layers):
                for variable_name in model.layer_variables_list:
                    variable = model.get_variable(layer_number, variable_name)
                    self.assertIsInstance(variable, tf.Variable)
                    self.assertIsNotNone(variable.eval())


class TestEstimator(tf.test.TestCase):

    def setUp(self):

        self._flags = get_default_flags()
        tf.reset_default_graph()

        topology = Topology()

        self.crocubot_model = CrocuBotModel(topology, self._flags, is_training=True)

    def test_forward_pass(self):

        estimator = Estimator(self.crocubot_model, self._flags)

        self.crocubot_model.build_layers_variables()
        data = np.ones(shape=(DEFAULT_BATCH_SIZE, DEFAULT_DEPTH, DEFAULT_N_SERIES, DEFAULT_TIMESTEPS), dtype=np.float32)
        output_signal = estimator.forward_pass(data)

        with self.test_session() as session:
            session.run(tf.global_variables_initializer())
            self.assertIsInstance(output_signal, tf.Tensor)
            value = output_signal.eval()
            self.assertEqual((DEFAULT_BATCH_SIZE, DEFAULT_DEPTH, DEFAULT_N_FORECASTS, DEFAULT_BINS), value.shape)

    def test_collate_multiple_passes(self):

        estimator = Estimator(self.crocubot_model, self._flags)

        self.crocubot_model.build_layers_variables()

        data = np.ones(shape=(DEFAULT_BATCH_SIZE, DEFAULT_DEPTH, DEFAULT_N_SERIES, DEFAULT_TIMESTEPS), dtype=np.float32)
        number_of_passes = 3
        stacked_output = estimator.collate_multiple_passes(data, number_of_passes)

        with self.test_session() as session:
            session.run(tf.global_variables_initializer())
            self.assertIsInstance(stacked_output, tf.Tensor)
            value = stacked_output.eval()
            expected_shape = (number_of_passes, DEFAULT_BATCH_SIZE, DEFAULT_DEPTH, DEFAULT_N_FORECASTS, DEFAULT_BINS)
            self.assertEqual(expected_shape, value.shape)

    def test_average_multiple_passes(self):

        estimator = Estimator(self.crocubot_model, self._flags)

        self.crocubot_model.build_layers_variables()

        data = np.ones(shape=(DEFAULT_BATCH_SIZE, DEFAULT_DEPTH, DEFAULT_N_SERIES, DEFAULT_TIMESTEPS), dtype=np.float32)
        number_of_passes = 3
        mean = estimator.average_multiple_passes(data, number_of_passes)

        with self.test_session() as session:
            session.run(tf.global_variables_initializer())

            self.assertIsInstance(mean, tf.Tensor)

            mean_value = mean.eval()

            expected_shape = (DEFAULT_BATCH_SIZE, DEFAULT_DEPTH, DEFAULT_N_FORECASTS, DEFAULT_BINS)
            self.assertEqual(expected_shape, mean_value.shape)
