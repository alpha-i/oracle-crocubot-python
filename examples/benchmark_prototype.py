# Example usage of crocubot
# Acts as a useful test that training & inference still works!

from timeit import default_timer as timer
import datetime
import logging
import numpy as np
import tensorflow as tf

from alphai_crocubot_oracle.data.classifier import BinDistribution
import alphai_crocubot_oracle.crocubot.evaluate as eval
from alphai_crocubot_oracle.crocubot.model import CrocuBotModel
import alphai_crocubot_oracle.flags as fl
import alphai_crocubot_oracle.iotools as io
import alphai_crocubot_oracle.topology as topo
from alphai_crocubot_oracle.crocubot import train as crocubot_train

from alphai_data_sources.data_sources import DataSourceGenerator
from alphai_data_sources.generator import BatchOptions, BatchGenerator
from alphai_time_series.performance_trials.performance import Metrics

data_source_generator = DataSourceGenerator()
batch_generator = BatchGenerator()
model_metrics = Metrics()

FLAGS = tf.app.flags.FLAGS
TIME_LIMIT = 600
D_TYPE = 'float32'


def run_timed_benchmark_mnist(series_name, do_training):

    topology = load_default_topology(series_name)
    batch_options = BatchOptions(batch_size=200,
                                 batch_number=0,
                                 train=do_training,
                                 dtype=D_TYPE)

    data_source = data_source_generator.make_data_source(series_name)
    _, labels = io.load_batch(batch_options, data_source)

    start_time = timer()
    execution_time = datetime.datetime.now()

    if do_training:
        crocubot_train.train(topology, series_name, execution_time)
    else:
        tf.reset_default_graph()
        model = CrocuBotModel(topology)
        model.build_layers_variables()

    mid_time = timer()
    train_time = mid_time - start_time
    print("Training complete.")

    metrics = evaluate_network(topology, series_name, bin_dist=None)
    eval_time = timer() - mid_time

    print('Metrics:')
    accuracy = print_MNIST_accuracy(metrics)
    print_time_info(train_time, eval_time)

    return accuracy, metrics


def print_time_info(train_time, eval_time):

    print('Training took', str.format('{0:.2f}', train_time), "seconds")
    print('Evaluation took', str.format('{0:.2f}', eval_time), "seconds")

    if train_time > TIME_LIMIT:
        print('** Training took ', str.format('{0:.2f}', train_time - TIME_LIMIT),
              ' seconds too long - DISQUALIFIED! **')


def run_timed_benchmark_time_series(series_name, flags, do_training=True):

    topology = load_default_topology(series_name)

    #  First need to establish bin edges using full training set
    template_sample_size = np.minimum(flags.n_training_samples_benchmark, 10000)

    batch_options = BatchOptions(batch_size=template_sample_size,
                                 batch_number=0,
                                 train=do_training,
                                 dtype=D_TYPE)

    data_source = data_source_generator.make_data_source(series_name)

    _, labels = io.load_batch(batch_options, data_source)

    bin_dist = BinDistribution(labels, topology.n_classification_bins)

    start_time = timer()

    execution_time = datetime.datetime.now()

    if do_training:
        crocubot_train.train(topology, series_name, execution_time, bin_edges=bin_dist.bin_edges)
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


def evaluate_network(topology, series_name, bin_dist):  # bin_dist not used in MNIST case

    # Get the test data
    batch_options = BatchOptions(batch_size=FLAGS.batch_size,
                                 batch_number=1,
                                 train=False,
                                 dtype=D_TYPE)

    data_source = data_source_generator.make_data_source(series_name)

    test_features, test_labels = io.load_batch(batch_options, data_source)
    save_file = io.load_file_name(series_name, topology)

    binned_outputs = eval.eval_neural_net(test_features, topology, save_file)
    n_samples = binned_outputs.shape[1]

    if series_name in {'mnist', 'mnist_reshaped'}:
        binned_outputs = np.mean(binned_outputs, axis=0)  # Average over passes
        predicted_indices = np.argmax(binned_outputs, axis=-1).flatten()
        true_indices = np.argmax(test_labels, axis=-1).flatten()

        print("Example forecasts:", binned_outputs[0:5, 0, :])
        print("Example outcomes", test_labels[0:5, 0, :])
        print("Total test samples:", n_samples)

        results = np.equal(predicted_indices, true_indices)
        forecasts = np.zeros(n_samples)
        p_success = []
        p_fail = []
        for i in range(n_samples):
            true_index = true_indices[i]
            forecasts[i] = binned_outputs[i, 0, true_index]

            if true_index == predicted_indices[i]:
                p_success.append(forecasts[i])
            else:
                p_fail.append(forecasts[i])

        log_likelihood_per_sample = np.mean(np.log(forecasts))
        median_probability = np.median(forecasts)

        metrics = {}
        metrics["results"] = results
        metrics["log_likelihood_per_sample"] = log_likelihood_per_sample
        metrics["median_probability"] = median_probability
        metrics["mean_p_success"] = np.mean(np.stack(p_success))
        metrics["mean_p_fail"] = np.mean(np.stack(p_fail))
        metrics["mean_p"] = np.mean(np.stack(forecasts))
        metrics["min_p_fail"] = np.min(np.stack(p_fail))

        return metrics

    else:
        estimated_means, estimated_covariance = eval.forecast_means_and_variance(binned_outputs, bin_dist)
        test_labels = np.squeeze(test_labels)

        model_metrics.evaluate_sample_performance(data_source, test_labels, estimated_means, estimated_covariance)


def load_default_topology(series_name):
    """The input and output layers must adhere to the dimensions of the features and labels.
    """

    layer_types = ['full', 'full', 'full', 'full']
    layer_heights = None,
    layer_widths = None,
    activation_functions = None
    n_features = 1

    if series_name == 'low_noise':
        n_input_series = 1
        n_timesteps = 100
        n_classification_bins = 12
        n_output_series = 1
    elif series_name == 'stochastic_walk':
        n_input_series = 10
        n_timesteps = 100
        n_classification_bins = 12
        n_output_series = 10
    elif series_name == 'mnist':
        if FLAGS.use_convolution:
            layer_types  = ['conv1d', 'full', 'full', 'full']
            layer_heights = [784, 400, 400, 10]
            layer_widths = [1, 1, 1, 1]
            activation_functions = ['linear', 'relu', 'relu', 'relu', 'linear']
        else:
            layer_types = ['full', 'full', 'full', 'full']
            layer_heights = [784, 400, 400, 10]
            layer_widths = [1, 1, 1, 1]
            activation_functions = ['linear', 'relu', 'relu', 'linear']
        n_input_series = 1
        n_timesteps = 784
        n_classification_bins = 10
        n_output_series = 1
    elif series_name == 'mnist_reshaped':
        if FLAGS.use_convolution:
            layer_types = ['conv3d', 'pool2d', 'conv3d', 'pool2d', 'full', 'full', 'full']
            layer_heights = [28, 28, 28, 400, 400, 400,  10]
            layer_widths = [28, 28, 28, 1, 1,  1, 1]

            activation_functions = ['linear', 'relu', 'relu', 'relu',  'relu', 'relu', 'linear']
        else:
            layer_types = ['full', 'full', 'full', 'full', 'full']
            layer_heights = [28, 28, 28, 400, 10]
            layer_widths = [28, 28, 28, 1, 1]
            activation_functions = ['linear', 'relu', 'relu', 'relu', 'linear']
        n_input_series = 1
        n_features = 28
        n_timesteps = 28
        n_classification_bins = 10
        n_output_series = 1
    else:
        raise NotImplementedError

    topology = topo.Topology(n_series=n_input_series, n_timesteps=n_timesteps,
                             n_forecasts=n_output_series,
                             n_classification_bins=n_classification_bins, layer_types=layer_types,
                             layer_heights=layer_heights, layer_widths=layer_widths,
                             activation_functions=activation_functions, n_features=n_features)

    return topology


def print_MNIST_accuracy(metrics):

    results = metrics["results"]

    total_tests = len(results)
    correct = np.sum(results)
    accuracy = correct / total_tests

    theoretical_max_log_likelihood_per_sample = np.log(0.5)*(1 - accuracy)

    print('MNIST accuracy of ', accuracy * 100, '%')
    print('Log Likelihood per sample of ', metrics["log_likelihood_per_sample"])
    print('Theoretical limit for given accuracy ', theoretical_max_log_likelihood_per_sample)
    print('Median probability assigned to true outcome:', metrics["median_probability"])
    print('Mean probability assigned to forecasts:', metrics["mean_p"])
    print('Mean probability assigned to successful forecast:', metrics["mean_p_success"])
    print('Mean probability assigned to unsuccessful forecast:', metrics["mean_p_fail"])
    print('Min probability assigned to unsuccessful forecast:', metrics["min_p_fail"])

    return accuracy


def run_mnist_test(train_path, tensorboard_log_path, method='GDO', use_full_train_set=True, reshape_to_2d=False):

    if use_full_train_set:
        n_training_samples = 60000
        n_epochs = 10
    else:
        n_training_samples = 500
        n_epochs = 100

    config = load_default_config()
    config["n_epochs"] = n_epochs
    config["learning_rate"] = 1e-3   # Use high learning rate for testing purposes
    config["cost_type"] = 'bayes'  # 'bayes'; 'softmax'; 'hellinger'
    config['batch_size'] = 200
    config['n_training_samples_benchmark'] = n_training_samples
    config['n_series'] = 1
    config['optimisation_method'] = method
    config['n_features_per_series'] = 784
    config['resume_training'] = False  # Make sure we start from scratch
    config['activation_functions'] = ['linear', 'selu', 'selu']
    config['tensorboard_log_path'] = tensorboard_log_path
    config['train_path'] = train_path
    config['model_save_path'] = train_path
    config['n_retrain_epochs'] = 5
    config['n_train_passes'] = 1
    config['n_eval_passes'] = 40
    config['use_convolution'] = False

    fl.set_training_flags(config)
    # this flag is only used in benchmark.
    tf.app.flags.DEFINE_integer('n_training_samples_benchmark', config['n_training_samples_benchmark'],
                                """Number of samples for benchmarking.""")
    FLAGS._parse_flags()
    print("Epochs to evaluate:", FLAGS.n_epochs)

    if reshape_to_2d:
        series_name = "mnist_reshaped"
    else:
        series_name = "mnist"

    return run_timed_benchmark_mnist(series_name=series_name, do_training=True)


def run_stochastic_test(train_path, tensorboard_log_path):
    config = load_default_config()

    config["n_epochs"] = 10   # -3 per sample after 10 epochs
    config["learning_rate"] = 3e-3   # Use high learning rate for testing purposes
    config["cost_type"] = 'bayes'  # 'bayes'; 'softmax'; 'hellinger'
    config['batch_size'] = 200
    config['n_training_samples_benchmark'] = 1000
    config['n_series'] = 10
    config['n_features_per_series'] = 100
    config['resume_training'] = False  # Make sure we start from scratch
    config['activation_functions'] = ['linear', 'selu', 'selu', 'selu']
    config["layer_heights"] = 200
    config["layer_widths"] = 1
    config['tensorboard_log_path'] = tensorboard_log_path
    config['train_path'] = train_path
    config['model_save_path'] = train_path
    config['n_retrain_epochs'] = 5
    config['use_convolution'] = False

    fl.set_training_flags(config)
    # this flag is only used in benchmark.
    tf.app.flags.DEFINE_integer('n_training_samples_benchmark', config['n_training_samples_benchmark'],
                                """Number of samples for benchmarking.""")
    FLAGS._parse_flags()
    print("Epochs to evaluate:", FLAGS.n_epochs)
    run_timed_benchmark_time_series(series_name='stochastic_walk', flags=FLAGS, do_training=True)


def load_default_config():
    configuration = {
        'data_transformation': {
            'feature_config_list': [
                {
                    'name': 'close',
                    'order': 'log-return',
                    'normalization': 'standard',
                    'nbins': 12,
                    'is_target': True,
                },
            ],
            'exchange_name': 'NYSE',
            'features_ndays': 10,
            'features_resample_minutes': 15,
            'features_start_market_minute': 60,
            'prediction_frequency_ndays': 1,
            'prediction_market_minute': 60,
            'target_delta_ndays': 1,
            'target_market_minute': 60,
        },
        'train_path': '/tmp/crocubot/',
        'covariance_method': 'NERCOME',
        'covariance_ndays': 9,
        'model_save_path': '/tmp/crocubot/',
        'd_type': D_TYPE,
        'tf_type': 32,
        'random_seed': 0,
        'predict_single_shares': False,

        # Training specific
        'n_epochs': 1,
        'n_training_samples_benchmark': 1000,
        'learning_rate': 2e-3,
        'batch_size': 100,
        'cost_type': 'bayes',
        'n_train_passes': 30,
        'n_eval_passes': 30,
        'resume_training': False,

        # Topology
        'n_series': 1,
        'n_features_per_series': 271,
        'n_forecasts': 1,
        'n_classification_bins': 12,
        'layer_heights': [200, 200, 200],
        'layer_widths': [1, 1, 1],
        'activation_functions': ['relu', 'relu', 'relu'],

        # Initial conditions
        'INITIAL_ALPHA': 0.8,
        'INITIAL_WEIGHT_UNCERTAINTY': 0.02,
        'INITIAL_BIAS_UNCERTAINTY': 0.02,
        'INITIAL_WEIGHT_DISPLACEMENT': 0.1,
        'INITIAL_BIAS_DISPLACEMENT': 0.1,
        'USE_PERFECT_NOISE': False,

        # Priors
        'double_gaussian_weights_prior': True,
        'wide_prior_std': 0.8,
        'narrow_prior_std': 0.001,
        'spike_slab_weighting': 0.5
    }

    return configuration


if __name__ == '__main__':

    logger = logging.getLogger('tipper')
    logger.addHandler(logging.StreamHandler())
    logging.basicConfig(level=logging.DEBUG)

    # change the following lines according to your machine
    train_path = '/tmp/'
    tensorboard_log_path = '/tmp/'

    # run_stochastic_test(train_path, tensorboard_log_path)
    run_mnist_test(train_path, tensorboard_log_path,  use_full_train_set=True)
