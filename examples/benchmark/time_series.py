import datetime
from timeit import default_timer as timer

import numpy as np
import tensorflow as tf
from alphai_data_sources.data_sources import DataSourceGenerator
from alphai_data_sources.generator import BatchOptions
from alphai_time_series.performance_trials.performance import Metrics

from alphai_crocubot_oracle import iotools as io
from alphai_crocubot_oracle.crocubot import train_with_datasource, evaluate as eval
from alphai_crocubot_oracle.crocubot.model import CrocuBotModel
from alphai_crocubot_oracle.data.classifier import BinDistribution
from alphai_crocubot_oracle.data.providers import TrainDataProviderForDataSource
from examples.helpers import D_TYPE, print_time_info, load_default_topology, FLAGS


def run_timed_benchmark_time_series(series_name, flags, do_training=True):

    topology = load_default_topology(series_name)

    #  First need to establish bin edges using full training set
    template_sample_size = np.minimum(flags.n_training_samples_benchmark, 10000)

    batch_options = BatchOptions(batch_size=template_sample_size,
                                 batch_number=0,
                                 train=do_training,
                                 dtype=D_TYPE)

    data_source = data_source_generator.make_data_source(series_name)

    _, labels = train_with_datasource.get_batch_from_generator(batch_options, data_source)

    bin_dist = BinDistribution(labels, topology.n_classification_bins)

    start_time = timer()

    execution_time = datetime.datetime.now()

    if do_training:
        train_with_datasource.train_with_datasource(topology, series_name, execution_time, bin_edges=bin_dist.bin_edges)
    else:
        tf.reset_default_graph()
        model = CrocuBotModel(topology)
        model.build_layers_variables()

    mid_time = timer()
    train_time = mid_time - start_time
    print("Training complete.")

    evaluate_network(topology, series_name, bin_dist)
    eval_time = timer() - mid_time

    print('Metrics:')
    print_time_info(train_time, eval_time)


data_source_generator = DataSourceGenerator()


def evaluate_network(topology, series_name, bin_dist):  # bin_dist not used in MNIST case
    data_provider = TrainDataProviderForDataSource(series_name, D_TYPE, FLAGS.batch_size * 2, False)

    test_features, test_labels = data_provider.get_batch(1, FLAGS.batch_size)

    save_file = io.build_check_point_filename(series_name, topology)

    binned_outputs = eval.eval_neural_net(test_features, topology, save_file)
    n_samples = binned_outputs.shape[1]

    model_metrics = Metrics()

    estimated_means, estimated_covariance = eval.forecast_means_and_variance(binned_outputs, bin_dist)
    test_labels = np.squeeze(test_labels)

    model_metrics.evaluate_sample_performance(data_provider.data_source, test_labels, estimated_means, estimated_covariance)
