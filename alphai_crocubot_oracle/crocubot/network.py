import tensorflow as tf


class LayeredNetwork(object):
    """

    """

    def __init__(self, topology, scope):
        self.verify_scope(scope)
        self.scope = scope
        self.topology = topology
        self.n_inputs = topology.n_inputs
        self.n_outputs = topology.n_outputs
        self.input = tf.placeholder(tf.float32, shape=[None, topology.n_inputs])
        self.output = tf.placeholder(tf.float32, shape=[None, topology.n_outputs])

    def verify_scope(self, scope):

        if not isinstance(scope, str):
            raise ValueError('The variable scope should be a string but got: {} '.format(scope))


# class BayesMLP(LayeredNetwork):
#     """
#     Initialises and provides access to the network's tensorflow variables
#     """
#
#     # To build the graph when instantiated
#     def __init__(self, parameters):
#         self.random_seed = 0
#         self.graph = tf.Graph()
#
#         with self.graph.as_default():
#             self.prediction = average_multiple_passes(x)
#             self.cost = cost.get_cost(prediction, truth)
#             self.optimizer = set_optimiser(parameters)
#
#         LayeredNetwork.__init__(self, layers, scope)
#         self.build_graph()
