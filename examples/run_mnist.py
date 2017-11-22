import tensorflow as tf
import logging

from alphai_crocubot_oracle import flags as fl
from examples.benchmark.mnist import run_timed_benchmark_mnist
from examples.helpers import load_default_config, FLAGS


def run_mnist_test(train_path, tensorboard_log_path, method='GDO', use_full_train_set=True):

    config = load_default_config()
    config["n_epochs"] = 10
    config["learning_rate"] = 1e-3   # Use high learning rate for testing purposes
    config["cost_type"] = 'bayes'  # 'bayes'; 'softmax'; 'hellinger'
    config['batch_size'] = 200
    config['n_series'] = 1
    config['optimisation_method'] = method
    config['n_features_per_series'] = 784
    config['resume_training'] = False  # Make sure we start from scratch
    config['activation_functions'] = ['linear', 'relu', 'relu']
    config['tensorboard_log_path'] = tensorboard_log_path
    config['train_path'] = train_path
    config['model_save_path'] = train_path
    config['n_retrain_epochs'] = 5
    config['n_train_passes'] = 1
    config['n_eval_passes'] = 10

    fl.build_tensorflow_flags(config)

    # this flag is only used in benchmark.
    tf.app.flags.DEFINE_integer('n_training_samples_benchmark', 60000, """Number of samples for benchmarking.""")
    tf.app.flags.DEFINE_integer('n_prediction_sample', 10000, """Number of samples for benchmarking.""")

    FLAGS._parse_flags()
    print("Epochs to evaluate:", FLAGS.n_epochs)

    run_timed_benchmark_mnist("mnist", FLAGS, do_training=True)


if __name__ == '__main__':

    logger = logging.getLogger('tipper')
    logger.addHandler(logging.StreamHandler())
    logging.basicConfig(level=logging.DEBUG)

    # change the following lines according to your machine
    train_path = '/tmp/'
    tensorboard_log_path = '/tmp/'

    run_mnist_test(train_path, tensorboard_log_path,  use_full_train_set=True)
