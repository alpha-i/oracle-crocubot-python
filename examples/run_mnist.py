import tensorflow as tf
import logging

from alphai_crocubot_oracle import flags as fl
from examples.benchmark.minst import run_timed_benchmark_mnist
from examples.helpers import load_default_config, FLAGS


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

    run_timed_benchmark_mnist(series_name="mnist", do_training=True)


if __name__ == '__main__':

    logger = logging.getLogger('tipper')
    logger.addHandler(logging.StreamHandler())
    logging.basicConfig(level=logging.DEBUG)

    # change the following lines according to your machine
    train_path = '/tmp/'
    tensorboard_log_path = '/tmp/'

    run_mnist_test(train_path, tensorboard_log_path,  use_full_train_set=True)
