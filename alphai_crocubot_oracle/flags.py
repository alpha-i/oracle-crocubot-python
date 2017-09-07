# Sets the crocubot hyperparameters as flags in tensorflow, based on a configuration dictionary
# Used by oracle.py

import tensorflow as tf
import argparse
FLAGS = tf.app.flags.FLAGS


def set_training_flags(config):
    """ Assigns flags based on entries in dictionary"""

    tf.flags._global_parser = argparse.ArgumentParser()

    tf.app.flags.DEFINE_string('log_path', config['tensorboard_path'], """Path for storing tensorboard log.""")

    tf.app.flags.DEFINE_string('d_type', config['d_type'], """Data type for numpy.""")
    tf.app.flags.DEFINE_integer('TF_TYPE', config['tf_type'], """Data type for tensorflow.""")
    tf.app.flags.DEFINE_integer('random_seed', 0, """Seed used to identify random noise realisiation.""")
    tf.app.flags.DEFINE_integer('n_classification_bins', config['n_classification_bins'], """How many bins to use for classification.""")
    tf.app.flags.DEFINE_string('model_save_path', config['model_save_path'], """Path to save graph.""")

    # Training specific
    tf.app.flags.DEFINE_integer('n_epochs', config['n_epochs'], """How many epochs to be used for training.""")
    tf.app.flags.DEFINE_integer('n_retrain_epochs', config['n_retrain_epochs'], """How many epochs to be used for re-training a previously stored model.""")
    tf.app.flags.DEFINE_integer('n_training_samples', config['n_training_samples'], """Total number of data samples to be used for training.""")
    tf.app.flags.DEFINE_float('learning_rate', config['learning_rate'], """Total number of data samples to be used for training.""")
    tf.app.flags.DEFINE_integer('batch_size', config['batch_size'], """Total number of data samples to be used for training.""")
    tf.app.flags.DEFINE_string('cost_type', config['cost_type'], """Total number of data samples to be used for training.""")
    tf.app.flags.DEFINE_integer('n_train_passes', config['n_train_passes'], """Number of passes to average over during training.""")
    tf.app.flags.DEFINE_integer('n_eval_passes', config['n_eval_passes'], """Number of passes to average over during evaluation.""")
    tf.app.flags.DEFINE_boolean('resume_training', config['resume_training'],
                                """Whether to set noise such that its mean and std are exactly the desired values""")
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
    FLAGS._parse_flags()


def dtype_from_tf_type(tf_dtype):
    if tf_dtype == tf.float64:
        return 'float64'
    elif tf_dtype == tf.float32:
        return 'float32'
    else:
        raise NotImplementedError
