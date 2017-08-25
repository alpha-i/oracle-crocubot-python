import tensorflow as tf
import numpy as np

from alphai_crocubot_oracle.crocubot.model import CrocuBotModel, Estimator
from alphai_crocubot_oracle.topology import Topology
from alphai_crocubot_oracle.flags import FLAGS, default as initialize_default_flags


class TestCrocuBotModel(tf.test.TestCase):

    def test_create_model(self):

        initialize_default_flags()

        layer_number = [
                {"activation_func": "relu", "trainable": False, "height": 20, "width": 10, "cell_height": 1},
                {"activation_func": "relu", "trainable": False, "height": 20, "width": 10, "cell_height": 1},
                {"activation_func": "linear", "trainable": False, "height": 20, "width": 10, "cell_height": 1}
        ]
        topology = Topology(layer_number)

        model = CrocuBotModel(topology, FLAGS)

        self.assertEqual(model.number_of_layers, 2)
        self.assertEqual(model.topology, topology)

        with self.test_session() as session:
            for layer_number in range(model.number_of_layers):
                for variable_name in model.layer_variables_list:
                    self.assertRaises(ValueError,
                                      model.get_variable,
                                      layer_number, variable_name
                              )

        with self.test_session() as session:

            model.build_layers_variables()
            session.run(tf.global_variables_initializer())

            self.assertEquals(len(session.run(tf.report_uninitialized_variables())), 0)

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

        data = np.ones(shape=(2, 200, 200), dtype=np.float32)
        output_signal = estimator.forward_pass(data)

        with self.test_session() as session:
            session.run(tf.global_variables_initializer())
            self.assertIsInstance(output_signal, tf.Tensor)
            value = output_signal.eval()
            print(value)

