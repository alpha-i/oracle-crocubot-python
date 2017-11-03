# Example usage of crocubot
# Acts as a useful test that training & inference still works!

import logging

import tensorflow as tf

import alphai_crocubot_oracle.flags as fl
from examples.benchmark.minst import run_timed_benchmark_mnist
from examples.benchmark.time_series import run_timed_benchmark_time_series
from examples.helpers import FLAGS, load_default_config


def run_mnist_test(train_path, tensorboard_log_path, method='Adam', use_full_train_set=True):

    if use_full_train_set:
        n_training_samples = 60000
        n_epochs = 2
    else:
        n_training_samples = 500
        n_epochs = 2

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

    fl.set_training_flags(config)
    # this flag is only used in benchmark.
    tf.app.flags.DEFINE_integer('n_training_samples_benchmark', config['n_training_samples_benchmark'],
                                """Number of samples for benchmarking.""")
    FLAGS._parse_flags()
    print("Epochs to evaluate:", FLAGS.n_epochs)

    return run_timed_benchmark_mnist(series_name="mnist", do_training=True)


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

    fl.set_training_flags(config)
    # this flag is only used in benchmark.
    tf.app.flags.DEFINE_integer('n_training_samples_benchmark', config['n_training_samples_benchmark'],
                                """Number of samples for benchmarking.""")
    FLAGS._parse_flags()
    print("Epochs to evaluate:", FLAGS.n_epochs)
    run_timed_benchmark_time_series(series_name='stochastic_walk', flags=FLAGS, do_training=True)


if __name__ == '__main__':

    logger = logging.getLogger('tipper')
    logger.addHandler(logging.StreamHandler())
    logging.basicConfig(level=logging.DEBUG)

    # change the following lines according to your machine
    train_path = '/tmp/'
    tensorboard_log_path = '/tmp/'

    # run_stochastic_test(train_path, tensorboard_log_path)
    run_mnist_test(train_path, tensorboard_log_path,  use_full_train_set=True)
