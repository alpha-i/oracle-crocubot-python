import tensorflow as tf

from alphai_crocubot_oracle import topology as topo

FLAGS = tf.app.flags.FLAGS
D_TYPE = 'float32'


def load_default_topology(series_name, tf_flags):
    """The input and output layers must adhere to the dimensions of the features and labels.
    """

    layer_types = ['full', 'full', 'full', 'full']
    layer_heights = None
    layer_widths = None
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
        if tf_flags.use_convolution:
            layer_types = ['conv1d', 'full', 'full', 'full']
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
        if tf_flags.use_convolution:
            layer_types = ['conv3d', 'pool2d', 'conv3d', 'pool2d', 'full', 'full', 'full']
            layer_heights = [28, 28, 28, 400, 400, 400, 10]
            layer_widths = [28, 28, 28, 1, 1, 1, 1]

            activation_functions = ['linear', 'relu', 'relu', 'relu', 'relu', 'relu', 'linear']
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


