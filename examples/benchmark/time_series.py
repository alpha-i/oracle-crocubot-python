import datetime

import numpy as np
import tensorflow as tf
from alphai_time_series.performance_trials.performance import Metrics

import alphai_crocubot_oracle.crocubot.evaluate as crocubot_eval
import alphai_crocubot_oracle.crocubot.train as crocubot_train

from alphai_crocubot_oracle.crocubot.helpers import TensorflowPath, TensorboardOptions
from alphai_crocubot_oracle.crocubot.model import CrocuBotModel
from alphai_crocubot_oracle.data.classifier import BinDistribution
from alphai_crocubot_oracle.data.providers import TrainDataProviderForDataSource
from alphai_crocubot_oracle.helpers import printtime, execute_and_get_duration

import examples.iotools as io
from examples.benchmark.helpers import print_time_info
from examples.helpers import D_TYPE, load_default_topology


def run_timed_benchmark_time_series(series_name, tf_flags, do_training=True):

    topology = load_default_topology(series_name, tf_flags)

    #  First need to establish bin edges using full training set
    n_train_samples = np.minimum(tf_flags.n_training_samples_benchmark, 10000)

    bin_distribution = _create_bin_distribution(series_name, n_train_samples, topology)
    batch_size = tf_flags.batch_size
    save_path = io.build_check_point_filename(series_name, topology, tf_flags)

    @printtime(message="Training {} with do_train: {}".format(series_name, int(do_training)))
    def _do_training():
        execution_time = datetime.datetime.now()
        if do_training:

            data_provider = TrainDataProviderForDataSource(
                series_name,
                D_TYPE,
                n_train_samples,
                batch_size,
                True,
                bin_distribution.bin_edges
            )

            tensorflow_path = TensorflowPath(save_path, tf_flags.model_save_path)
            tensorboard_options = TensorboardOptions(tf_flags.tensorboard_log_path,
                                                     tf_flags.learning_rate,
                                                     batch_size,
                                                     execution_time
                                                     )
            crocubot_train.train(topology,
                                 data_provider,
                                 tensorflow_path,
                                 tensorboard_options,
                                 tf_flags
                                 )
        else:
            tf.reset_default_graph()
            model = CrocuBotModel(topology)
            model.build_layers_variables()

    train_time, _ = execute_and_get_duration(_do_training)

    print("Training complete.")

    eval_time, _ = execute_and_get_duration(evaluate_network, topology, series_name, batch_size,
                                            save_path, bin_distribution, tf_flags)

    print('Metrics:')
    print_time_info(train_time, eval_time)


def _create_bin_distribution(series_name, n_training_samples, topology):
    data_provider = TrainDataProviderForDataSource(series_name, D_TYPE, n_training_samples, n_training_samples, True)
    train_data = data_provider.get_batch(0)

    return BinDistribution(train_data.labels, topology.n_classification_bins)


@printtime(message="Evaluation of Stocastic Serie")
def evaluate_network(topology, series_name, batch_size, save_path, bin_dist, tf_flags):

    n_training_samples = batch_size * 2
    data_provider = TrainDataProviderForDataSource(series_name, D_TYPE, n_training_samples, batch_size, False)

    test_features, test_labels = data_provider.get_batch(1)

    binned_outputs = crocubot_eval.eval_neural_net(test_features, topology, tf_flags, save_path)

    estimated_means, estimated_covariance = crocubot_eval.forecast_means_and_variance(
        binned_outputs, bin_dist, tf_flags)
    test_labels = np.squeeze(test_labels)

    model_metrics = Metrics()
    model_metrics.evaluate_sample_performance(
        data_provider.data_source,
        test_labels,
        estimated_means,
        estimated_covariance
    )
