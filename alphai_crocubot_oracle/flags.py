# Sets the crocubot hyperparameters as flags in tensorflow, based on a configuration dictionary
# Used by oracle.py

import tensorflow as tf
import argparse

DEFAULT_RANDOM_SEED = 42


def build_tensorflow_flags(config):
    """ Assigns flags based on entries in dictionary"""

    tf.flags._global_parser = argparse.ArgumentParser()

    config = update_config_defaults(config)

    random_seed = config.get('random_seed', DEFAULT_RANDOM_SEED)

    tf.app.flags.DEFINE_boolean('do_batch_norm', config['do_batch_norm'],
                                """Whether to use batch normalisation. """)

    tf.app.flags.DEFINE_boolean('apply_temporal_suppression', config['apply_temporal_suppression'],
                                """Whether to penalise data which is further in the past. """)

    tf.app.flags.DEFINE_boolean('partial_retrain', config['partial_retrain'],
                                """Whether to retrain all layers or just the fully connected ones. """)

    tf.app.flags.DEFINE_boolean('use_convolution', config['use_convolution'],
                                """Whether to set the first layer to a convolutional layer""")

    tf.app.flags.DEFINE_boolean('predict_single_shares', config['predict_single_shares'],
                                """Whether the network predicts one share at a time.""")

    tf.app.flags.DEFINE_string('tensorboard_log_path', config['tensorboard_log_path'], """Path for storing tensorboard log.""")
    tf.app.flags.DEFINE_string('d_type', config['d_type'], """Data type for numpy.""")


    tf.app.flags.DEFINE_integer('TF_TYPE', config['tf_type'], """Data type for tensorflow.""")
    tf.app.flags.DEFINE_integer('random_seed', random_seed, """Seed used to identify random noise realisiation.""")
    tf.app.flags.DEFINE_integer('n_classification_bins', config['n_classification_bins'], """How many bins to use for classification.""")
    tf.app.flags.DEFINE_string('model_save_path', config['model_save_path'], """Path to save graph.""")
    tf.app.flags.DEFINE_string('optimisation_method', config['optimisation_method'], """Algorithm for training""")

    # Training specific
    tf.app.flags.DEFINE_integer('n_epochs', config['n_epochs'], """How many epochs to be used for training.""")
    tf.app.flags.DEFINE_integer('n_retrain_epochs', config['n_retrain_epochs'], """How many epochs to be used for re-training a previously stored model.""")
    tf.app.flags.DEFINE_float('learning_rate', config['learning_rate'], """Total number of data samples to be used for training.""")
    tf.app.flags.DEFINE_float('retrain_learning_rate', config['retrain_learning_rate'], """Total number of data samples to be used for training.""")

    tf.app.flags.DEFINE_integer('batch_size', config['batch_size'], """Total number of data samples to be used for training.""")
    tf.app.flags.DEFINE_string('cost_type', config['cost_type'], """Total number of data samples to be used for training.""")
    tf.app.flags.DEFINE_integer('n_train_passes', config['n_train_passes'], """Number of passes to average over during training.""")
    tf.app.flags.DEFINE_float('noise_amplitude', config['noise_amplitude'], """Additive noise for features. Defaults to zero. """)
    tf.app.flags.DEFINE_boolean('resume_training', config['resume_training'],
                                """Whether to set noise such that its mean and std are exactly the desired values""")

    # Eval specific
    tf.app.flags.DEFINE_integer('n_networks', config['n_networks'], """Number of networks to evaluate.""")
    tf.app.flags.DEFINE_integer('n_eval_passes', config['n_eval_passes'], """Number of passes to average over during evaluation.""")

    # Initial conditions
    tf.app.flags.DEFINE_float('INITIAL_ALPHA', config['INITIAL_ALPHA'], """Prior on weights.""")
    tf.app.flags.DEFINE_float('INITIAL_WEIGHT_UNCERTAINTY', config['INITIAL_WEIGHT_UNCERTAINTY'], """Initial standard deviation on weights.""")
    tf.app.flags.DEFINE_float('INITIAL_BIAS_UNCERTAINTY', config['INITIAL_BIAS_UNCERTAINTY'], """Initial standard deviation on bias.""")
    tf.app.flags.DEFINE_float('INITIAL_WEIGHT_DISPLACEMENT', config['INITIAL_WEIGHT_DISPLACEMENT'], """Initial offset on weight distributions.""")
    tf.app.flags.DEFINE_float('INITIAL_BIAS_DISPLACEMENT', config['INITIAL_BIAS_DISPLACEMENT'], """Initial offset on bias distributions.""")
    tf.app.flags.DEFINE_boolean('USE_PERFECT_NOISE', config['USE_PERFECT_NOISE'],
                                """Whether to set noise such that its mean and std are exactly the desired values""")
    # Priors
    tf.app.flags.DEFINE_boolean('double_gaussian_weights_prior', config['double_gaussian_weights_prior'],
                                """Whether to impose a double Gaussian prior.""")
    tf.app.flags.DEFINE_float('wide_prior_std', config['wide_prior_std'], """Initial standard deviation on weights.""")
    tf.app.flags.DEFINE_float('narrow_prior_std', config['narrow_prior_std'], """Initial standard deviation on weights.""")
    tf.app.flags.DEFINE_float('spike_slab_weighting', config['spike_slab_weighting'], """Initial standard deviation on weights.""")

    tf.app.flags.FLAGS._parse_flags()

    return tf.app.flags.FLAGS

def update_config_defaults(config):
    """

    :param config:
    :return:
    """

    if 'optimisation_method' not in config:
        config['optimisation_method'] = 'Adam'

    if 'use_convolution' not in config:
        config['use_convolution'] = 'False'

    if 'partial_retrain' not in config:
        config['partial_retrain'] = 'False'

    if 'apply_temporal_suppression' not in config:
        config['apply_temporal_suppression'] = 'True'

    if 'do_batch_norm' not in config:
        config['do_batch_norm'] = 'True'

    if 'noise_amplitude' not in config:
        config['noise_amplitude'] = 0

    if 'retrain_learning_rate' not in config:
        config['retrain_learning_rate'] = config['learning_rate']

    if 'n_networks' not in config:
        config['n_networks'] = 1

    return config


def dtype_from_tf_type(tf_dtype):
    if tf_dtype == tf.float64:
        return 'float64'
    elif tf_dtype == tf.float32:
        return 'float32'
    else:
        raise NotImplementedError
