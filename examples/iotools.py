import os
import tensorflow as tf

from alphai_data_sources.generator import BatchGenerator


def build_check_point_filename(series_name, topology, tf_flags):
    """ File used for storing the network parameters.

    :param str series_name: Identify the data on which the network was trained: MNIST, low_noise, randomwalk, etc
    :param Topology topology: Info on network shape
    :param tf_flags:

    :return:
    """

    depth_string = str(topology.n_layers)
    breadth_string = str(topology.n_timesteps)
    series_string = str(topology.n_series)

    bitstring = str(tf_flags.TF_TYPE)

    file_name = "{}model_{}_{}_{}x{}.ckpt".format(bitstring[-2:], series_name, series_string, depth_string,
                                                  breadth_string)
    return os.path.join(tf_flags.model_save_path, file_name)
