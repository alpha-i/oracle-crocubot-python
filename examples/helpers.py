import tensorflow as tf

from alphai_crocubot_oracle import topology as topo

FLAGS = tf.app.flags.FLAGS
TIME_LIMIT = 600
D_TYPE = 'float32'


def print_time_info(train_time, eval_time):

    print('Training took', str.format('{0:.2f}', train_time), "seconds")
    print('Evaluation took', str.format('{0:.2f}', eval_time), "seconds")

    if train_time > TIME_LIMIT:
        print('** Training took ', str.format('{0:.2f}', train_time - TIME_LIMIT),
              ' seconds too long - DISQUALIFIED! **')


def load_default_topology(series_name):
    """The input and output layers must adhere to the dimensions of the features and labels.
    """

    if series_name == 'low_noise':
        n_input_series = 1
        n_features_per_series = 100
        n_classification_bins = 12
        n_output_series = 1
    elif series_name == 'stochastic_walk':
        n_input_series = 10
        n_features_per_series = 100
        n_classification_bins = 12
        n_output_series = 10
    elif series_name == 'mnist':
        n_input_series = 1
        n_features_per_series = 784
        n_classification_bins = 10
        n_output_series = 1
    else:
        raise NotImplementedError

    return topo.Topology(layers=None, n_series=n_input_series, n_features_per_series=n_features_per_series, n_forecasts=n_output_series,
                         n_classification_bins=n_classification_bins)


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
        'INITIAL_WEIGHT_UNCERTAINTY': 0.01,
        'INITIAL_BIAS_UNCERTAINTY': 0.001,
        'INITIAL_WEIGHT_DISPLACEMENT': 0.001,
        'INITIAL_BIAS_DISPLACEMENT': 0.0001,
        'USE_PERFECT_NOISE': False,

        # Priors
        'double_gaussian_weights_prior': True,
        'wide_prior_std': 0.8,
        'narrow_prior_std': 0.001,
        'spike_slab_weighting': 0.5
    }

    return configuration


