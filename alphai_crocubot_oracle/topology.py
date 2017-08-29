# Defines the layout of the network
# Used by oracle, crocubot_model, and crocubot_train

import tensorflow as tf
import alphai_crocubot_oracle.tensormaths as tm

ACTIVATION_FN_LINEAR = "linear"
ACTIVATION_FN_SELU = "selu"
ACTIVATION_FN_RELU = "relu"

ALLOWED_ACTIVATION_FN = [ACTIVATION_FN_RELU, ACTIVATION_FN_SELU, ACTIVATION_FN_LINEAR]

DEFAULT_N_SERIES = 3
DEFAULT_FEAT_PER_SERIES = 10
DEFAULT_BINS = 10
DEFAULT_N_FORECASTS = 3
DEFAULT_HEIGHTS = [DEFAULT_N_SERIES, 10, DEFAULT_N_FORECASTS]
DEFAULT_WIDTHS = [DEFAULT_FEAT_PER_SERIES, 10, DEFAULT_BINS]
DEFAULT_ACT_FUNCTIONS = ['linear', 'relu', 'linear']


class Topology(object):
    """
    A class for containing the information that defines the topology of the neural network.
    Run checks on the user input to verify that it defines a valid topology.
    """

    def __init__(self, layers=None, n_series=DEFAULT_N_SERIES, n_features_per_series=DEFAULT_FEAT_PER_SERIES, n_forecasts=DEFAULT_N_FORECASTS,
                 n_classification_bins=DEFAULT_BINS, layer_heights=DEFAULT_HEIGHTS, layer_widths=DEFAULT_WIDTHS, activation_functions=DEFAULT_ACT_FUNCTIONS):
        """
        Following info is required to construct a topology object
        :param layers: Full list of layers can be provided, or:
        :param n_series:
        :param n_features_per_series:
        :param n_forecasts:
        :param n_classification_bins:
        :param layer_heights:
        :param layer_widths:
        :param activation_functions:
        """

        if layers is None:
            layers = self._build_layers(layer_heights, layer_widths, activation_functions)
            # FIXME Short term hack to ensure consistency - the following four lines should probably be assertions
            layers[0]["width"] = n_features_per_series
            layers[0]["height"] = n_series
            layers[-1]["height"] = n_forecasts
            layers[-1]["width"] = n_classification_bins

        self._verify_layers(layers)
        self.layers = layers
        self.n_series = n_series
        self.n_layers = len(layers) - 1  # n layers of neurons are connected by n-1 sets of weights
        self.n_features_per_series = n_features_per_series
        self.n_forecasts = n_forecasts
        self.n_classification_bins = n_classification_bins

    def _verify_layers(self, layers):
        """
        A function that checks each layer to ensure that it is valid i.e., expected activation function, trainable
        flag, etc.
        :param layers:
        :return: None
        """
        for i, layer in enumerate(layers):

            if layer["activation_func"] not in ALLOWED_ACTIVATION_FN:
                raise ValueError('Unexpected activation function ' + str(layer["activation_func"]))

            for key in ['height', 'width']:
                x = layer[key]
                if not (isinstance(x, int) and x > 0):
                    raise ValueError(
                        'Layer {} {} should be a positive integer'.format(i, key)
                    )

            if not isinstance(layer["trainable"], bool):
                raise ValueError('Layer {} trainable should be a boolean'.format(i))

    def get_cell_shape(self, layer_number):
        """
        returns the shape of the cells in a layer specified by the layer number
        :param layer_number: int
        :return: [int, int]
        """

        cell_height = self.layers[layer_number]["cell_height"]
        cell_width = self.layers[layer_number]["width"]

        return [cell_height, cell_width]

    def get_weight_shape(self, layer_number):

        if layer_number >= self.n_layers:
            raise ValueError('layer_number should be strictly less the number of layers')

        input_height = self.layers[layer_number]["height"]
        input_width = self.layers[layer_number]["width"]

        output_height = self.layers[layer_number + 1]["height"]
        output_width = self.layers[layer_number + 1]["width"]

        weight_shape = [input_height, input_width, output_height, output_width]

        return weight_shape

    def get_bias_shape(self, layer_number):
        """
        returns the shape of the biases in a layer specified by layer number as an array
        :param layer_number:
        :return:
        """
        if layer_number >= self.n_layers:
            raise ValueError('layer_number should be strictly less the number of layers')

        height = self.layers[layer_number + 1]["height"]
        width = self.layers[layer_number + 1]["width"]

        bias_shape = [height, width]

        return bias_shape

    def get_activation_function(self, layer_number):

        function_name = self.layers[layer_number + 1]["activation_func"]

        if function_name == 'linear':
            return lambda x: x
        elif function_name == 'selu':
            return tm.selu
        elif function_name == 'relu':
            return tf.nn.relu
        elif function_name == 'kelu':
            return tm.kelu
        else:
            raise NotImplementedError

    def _build_layers(self, layer_heights, layer_widths, activation_functions):
        """
        :param activation_functions:
        :param n_series:
        :param n_features_per_series:
        :param n_forecasts:
        :param n_classification_bins:
        :param layer_heights:
        :param layer_widths:
        :return:
        """

        layers = []
        n_layers = len(activation_functions)

        for i in range(n_layers):
            layer = {}
            layer["activation_func"] = activation_functions[i]
            layer["trainable"] = True  # Just hardcode for now, will be configurable in future
            layer["height"] = layer_heights[i]
            layer["width"] = layer_widths[i]
            layer["cell_height"] = 1  # Just hardcode for now, will be configurable in future

            layers.append(layer)

        return layers
