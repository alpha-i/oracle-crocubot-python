import tensorflow as tf
import logging

from alphai_crocubot_oracle import flags as fl
from examples.benchmark.time_series import run_timed_benchmark_time_series
from examples.helpers import load_default_config, FLAGS


def run_stochastic_test(train_path, tensorboard_log_path):
    config = load_default_config()

    config["n_epochs"] = 10  # -3 per sample after 10 epochs
    config["learning_rate"] = 3e-3  # Use high learning rate for testing purposes
    config["cost_type"] = 'bayes'  # 'bayes'; 'softmax'; 'hellinger'
    config['batch_size'] = 200
    config['n_training_samples_benchmark'] = 1000
    config['n_series'] = 10
    config['n_features_per_series'] = 100
    config['resume_training'] = False  # Make sure we start from scratch
    config['activation_functions'] = ['linear', 'selu', 'selu', 'selu']
    config["layer_heights"] = 4
    config["layer_widths"] = 1
    config['tensorboard_log_path'] = tensorboard_log_path
    config['train_path'] = train_path
    config['model_save_path'] = train_path
    config['n_retrain_epochs'] = 5
    config['use_convolution'] = False

    fl.build_tensorflow_flags(config)

    # this flag is only used in benchmark.
    tf.app.flags.DEFINE_integer('n_training_samples_benchmark', 1000, """Number of samples for benchmarking.""")
    tf.app.flags.DEFINE_integer('n_prediction_sample', 1000, """Number of samples for benchmarking.""")

    FLAGS._parse_flags()
    print("Epochs to evaluate:", FLAGS.n_epochs)
    run_timed_benchmark_time_series('stochastic_walk', tf_flags=FLAGS, do_training=True)


if __name__ == '__main__':

    logger = logging.getLogger('tipper')
    logger.addHandler(logging.StreamHandler())
    logging.basicConfig(level=logging.DEBUG)

    train_path = '/tmp/'
    tensorboard_log_path = '/tmp/'

    run_stochastic_test(train_path, tensorboard_log_path)
