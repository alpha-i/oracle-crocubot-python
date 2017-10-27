# TODO this module is not used anywhere. move or remove it.

MNIST_STANDARD_ARCH = [{"activation_func": "relu", "trainable": False, "height": 28, "width": 28, "cell_height": 1},
                       {"activation_func": "selu", "trainable": False, "height": 20, "width": 10, "cell_height": 1},
                       {"activation_func": "linear", "trainable": False, "height": 20, "width": 10, "cell_height": 1}]


STOCKS_STANDARD_ARCH = [{"activation_func": "relu", "trainable": False, "height": 20, "width": 10, "cell_height": 1},
                        {"activation_func": "relu", "trainable": False, "height": 20, "width": 10, "cell_height": 1},
                        {"activation_func": "linear", "trainable": False, "height": 20, "width": 10, "cell_height": 1}]
