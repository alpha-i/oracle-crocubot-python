import tensorflow as tf
import numpy as np

from alphai_crocubot_oracle.crocubot.model import CrocuBotModel, Estimator
from alphai_crocubot_oracle.topology import Topology
from alphai_crocubot_oracle.flags import FLAGS
from tests.helpers import default as initialize_default_flags


class TestCrocuBotModel(tf.test.TestCase):

    def test_create_model(self):

        initialize_default_flags()

        layer_dict = [
            {"activation_func": "relu", "trainable": False, "height": 20, "width": 10, "cell_height": 1},
            {"activation_func": "relu", "trainable": False, "height": 20, "width": 10, "cell_height": 1},
            {"activation_func": "linear", "trainable": False, "height": 20, "width": 10, "cell_height": 1}
        ]
        topology = Topology(layer_dict)

        model = CrocuBotModel(topology, FLAGS)

        self.assertEqual(model.number_of_layers, 2)
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

        initialize_default_flags()
        tf.reset_default_graph()

        layer_number = [
            {"activation_func": "relu", "trainable": False, "height": 20, "width": 10, "cell_height": 1},
            {"activation_func": "relu", "trainable": False, "height": 20, "width": 10, "cell_height": 1},
            {"activation_func": "linear", "trainable": False, "height": 20, "width": 10, "cell_height": 1}
        ]
        topology = Topology(layer_number)

        self.crocubot_model = CrocuBotModel(topology, FLAGS)

    def test_forward_pass(self):

        estimator = Estimator(self.crocubot_model, FLAGS)

        self.crocubot_model.build_layers_variables()

        data = np.ones(shape=(20, 10), dtype=np.float32)
        output_signal = estimator.forward_pass(data)

        with self.test_session() as session:
            session.run(tf.global_variables_initializer())
            self.assertIsInstance(output_signal, tf.Tensor)
            value = output_signal.eval()
            self.assertEqual((20, 10), value.shape)

    def test_collate_multiple_passes(self):

        estimator = Estimator(self.crocubot_model, FLAGS)

        self.crocubot_model.build_layers_variables()

        data = np.ones(shape=(20, 10), dtype=np.float32)
        number_of_passes = 3
        stacked_output = estimator.collate_multiple_passes(data, number_of_passes)

        with self.test_session() as session:
            session.run(tf.global_variables_initializer())
            self.assertIsInstance(stacked_output, tf.Tensor)
            value = stacked_output.eval()
            expected_shape = (number_of_passes, 20, 10)
            self.assertEqual(expected_shape, value.shape)

    def test_average_multiple_passes(self):

        estimator = Estimator(self.crocubot_model, FLAGS)

        self.crocubot_model.build_layers_variables()

        data = np.ones(shape=(20, 10), dtype=np.float32)
        number_of_passes = 3
        mean = estimator.average_multiple_passes(data, number_of_passes)

        with self.test_session() as session:
            session.run(tf.global_variables_initializer())

            self.assertIsInstance(mean, tf.Tensor)

            mean_value = mean.eval()

            expected_shape = (20, 10)
            self.assertEqual(expected_shape, mean_value.shape)
