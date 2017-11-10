# Defines the layout of the network
# Used by oracle, crocubot_model, and crocubot_train

import tensorflow as tf

import alphai_crocubot_oracle.tensormaths as tm

ACTIVATION_FN_LINEAR = "linear"
ACTIVATION_FN_SELU = "selu"
ACTIVATION_FN_RELU = "relu"
LAYER_FULL = 'full'
LAYER_CONV_3D = 'conv3d'
LAYER_POOL = 'pool2d'

ALLOWED_ACTIVATION_FN = [ACTIVATION_FN_RELU, ACTIVATION_FN_SELU, ACTIVATION_FN_LINEAR]
ALLOWED_LAYER_TYPES = [LAYER_FULL, LAYER_CONV_3D, LAYER_POOL]

DEFAULT_N_SERIES = 28
DEFAULT_TIMESTEPS = 28
DEFAULT_BINS = 10
DEFAULT_N_FORECASTS = 3
DEFAULT_HIDDEN_LAYERS = 2
DEFAULT_HEIGHT = 400  # NB this is the dimension which gets shuffled
DEFAULT_WIDTH = 1  # NB noise in this dimension is not shuffled
DEFAULT_ACT_FUNCTION = 'relu'
DEFAULT_LAYER_TYPE = 'full'
DEFAULT_N_KERNELS = 4
DEFAULT_N_FEATURES = 1
DEFAULT_DEPTH = 1
DEFAULT_N_OUTPUT_SERIES = 1


class Topology(object):
    """
    A class for containing the information that defines the topology of the neural network.
    Run checks on the user input to verify that it defines a valid topology.
    """

    def __init__(self, n_series=DEFAULT_N_SERIES, n_timesteps=DEFAULT_TIMESTEPS,
                 n_forecasts=DEFAULT_N_FORECASTS, n_classification_bins=DEFAULT_BINS, layer_heights=None,
                 layer_widths=None, layer_depths=None, activation_functions=None, layer_types=None, n_features=1):
        """
        Following info is required to construct a topology object
        :param n_series:
        :param n_timesteps:
        :param n_forecasts:
        :param n_classification_bins:
        :param layer_heights:
        :param layer_widths:
        :param activation_functions:
        """

        if layer_heights is None:
            assert layer_widths is None and activation_functions is None and layer_depths is None
            layer_depths, layer_heights, layer_widths, activation_functions = \
                self.get_default_layers(DEFAULT_HIDDEN_LAYERS)
        else:
            if layer_depths is None:
                layer_depths = [DEFAULT_DEPTH]*len(layer_heights)
            else:
                assert len(layer_depths) == len(layer_heights), "Length of depths array does not match height array"
            assert len(layer_widths) == len(layer_heights), "Length of widths array does not match height array"
            assert len(activation_functions) == len(layer_heights), "Length of act fns does not match height array"

        layers = self._build_layers(layer_depths, layer_heights, layer_widths, activation_functions, layer_types)
        # FIXME Short term hack to ensure consistency - the following four lines should probably be assertions
        layers[0]["depth"] = n_series
        layers[0]["height"] = n_timesteps
        layers[0]["width"] = n_features

        layers[-1]["height"] = n_forecasts
        layers[-1]["width"] = n_classification_bins

        self._verify_layers(layers)
        self.layers = layers
        self.n_series = n_series
        self.n_layers = len(layers) - 1  # n layers of neurons are connected by n-1 sets of weights
        self.n_timesteps = n_timesteps
        self.n_features = n_features
        self.n_forecasts = n_forecasts
        self.n_classification_bins = n_classification_bins
        self.n_parameters = self._calculate_number_of_parameters(layers)
        self.n_kernels = DEFAULT_N_KERNELS

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

            if layer["type"] not in ALLOWED_LAYER_TYPES:
                raise ValueError('Unexpected layer type ' + str(layer["type"]))

            for key in ['height', 'width']:
                x = layer[key]
                if not (isinstance(x, int) and x > 0):
                    raise ValueError(
                        'Layer {} {} should be a positive integer'.format(i, key)
                    )

            if not isinstance(layer["trainable"], bool):
                raise ValueError('Layer {} trainable should be a boolean'.format(i))

    def _calculate_number_of_parameters(self, layers):
        """ Returns total number of connections, assuming layers are fully connected"""

        n_parameters = 0
        for i in range(self.n_layers):
            j = i + 1
            n_parameters += layers[i]["width"] * layers[i]["height"] * layers[j]["height"] * layers[j]["width"]

        return n_parameters

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
        input_depth = self.layers[layer_number]["depth"]

        output_height = self.layers[layer_number + 1]["height"]
        output_width = self.layers[layer_number + 1]["width"]
        output_depth = self.layers[layer_number + 1]["depth"]

        weight_shape = [input_depth, input_height, input_width, output_depth, output_height, output_width]

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
        output_depth = self.layers[layer_number + 1]["depth"]
        bias_shape = [output_depth, height, width]

        return bias_shape

    def get_layer_type(self, layer_number):

        return self.layers[layer_number]["type"]

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

    def _build_layers(self, layer_depths, layer_heights, layer_widths, activation_functions, layer_types=None):
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
            layer["cell_height"] = 1  # Just hardcode for now, will be configurable in future
            layer["depth"] = layer_depths[i]
            layer["height"] = layer_heights[i]
            layer["width"] = layer_widths[i]

            if layer_types is None:
                layer["type"] = DEFAULT_LAYER_TYPE
            else:
                layer["type"] = layer_types[i]

            if i > 0:  # Enforce consistent dimensions for subsequent layers
                prev_layer = layers[i - 1]
                previous_layer_type = prev_layer["type"]

                if previous_layer_type == 'pool2d':  # Pooling will rescale size of last layer
                    layer["height"] = int(prev_layer['height'] / 2)
                    layer["width"] = int(prev_layer['width'] / 2)
                elif previous_layer_type in {'conv2d', 'conv1d', 'conv3d'}:
                    # This will depend on choice of padding. Default for now is same, so easier.
                    layer["depth"] = int(prev_layer["depth"])
                    layer["height"] = int(prev_layer["height"])
                    layer["width"] = int(prev_layer["width"]) * DEFAULT_N_KERNELS

            layers.append(layer)

        return layers

    @staticmethod
    def get_default_layers(n_hidden_layers):
        """ Compiles the list of layer heights, widths and activation funcs to be used if none are provided

        :return:
        """

        layer_depths = [DEFAULT_N_FEATURES] + [DEFAULT_DEPTH] * n_hidden_layers + [DEFAULT_N_OUTPUT_SERIES]
        layer_heights = [DEFAULT_N_SERIES] + [DEFAULT_HEIGHT] * n_hidden_layers + [DEFAULT_N_FORECASTS]
        layer_widths = [DEFAULT_TIMESTEPS] + [DEFAULT_WIDTH] * n_hidden_layers + [DEFAULT_BINS]

        activation_functions = ['linear'] + [DEFAULT_ACT_FUNCTION] * n_hidden_layers + ['linear']

        return layer_depths, layer_heights, layer_widths, activation_functions

    def get_network_input_shape(self):
        """ Returns the required shape of input data.

        :return: 3D Numpy array
        """

        input_layer = self.layers[0]
        input_shape = (input_layer["depth"], input_layer["height"], input_layer["width"])

        return input_shape
