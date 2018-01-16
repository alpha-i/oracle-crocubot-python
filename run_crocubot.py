import os
import datetime
import pytz
import yaml

import logging

os.environ['MKL_THREADING_LAYER'] = 'GNU'

from alphai_delphi.controller import ControllerConfiguration, Controller
from alphai_delphi.data_source.synthetic_data_source import SyntheticDataSource
from alphai_delphi.oracle.oracle_configuration import OracleConfiguration
from alphai_delphi.performance.performance import OraclePerformance
from alphai_delphi.scheduler import Scheduler

from alphai_crocubot_oracle.oracle import CrocubotOracle

logging.basicConfig(level=logging.DEBUG)

OUTPUT_DIR = './result'

exchange = "NYSE"
simulation_start = datetime.datetime(2016, 1, 1, tzinfo=pytz.utc)
simulation_end = datetime.datetime(2016, 3, 1, tzinfo=pytz.utc)
N_ASSETS = 10

oracle_config = {

    'data_transformation': {
        'feature_config_list': [
            {
                'is_target': False,
                'length': 100,
                'name': 'close',
                'normalization': 'standard',
                'resolution': 10,
                'transformation': {'name': 'log-return'}
            },
            {
                'is_target': False,
                'length': 100,
                'name': 'close',
                'normalization': 'standard',
                'resolution': 100,
                'transformation': {'name': 'log-return'}
            },
            {
                'is_target': True,
                'length': 100,
                'name': 'close',
                'normalization': 'standard',
                'resolution': 1440,
                'transformation': {'name': 'log-return'}
            }
        ],
        'exchange_name': exchange,
        'features_ndays': 100,
        'features_resample_minutes': 10,
        'fill_limit': 0,
        'predict_the_market_close': True
    },

    'universe': {
        'avg_function': 'median',
        'dropna': False,
        'method': 'liquidity_day',
        'nassets': N_ASSETS,
        'ndays_window': 30,
        'update_frequency': 'monthly'
    },

    'INITIAL_ALPHA': 0.2,
    'INITIAL_BIAS_DISPLACEMENT': 0.1,
    'INITIAL_BIAS_UNCERTAINTY': 0.01,
    'INITIAL_WEIGHT_DISPLACEMENT': 0.1,
    'INITIAL_WEIGHT_UNCERTAINTY': 0.01,
    'USE_PERFECT_NOISE': False,
    'activation_functions': ['relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'relu', 'linear'],
    'apply_temporal_suppression': False,
    'batch_size': 300,
    'classify_per_series': False,
    'cost_type': 'bayes',
    'covariance_method': 'Ledoit',
    'covariance_ndays': 100,
    'd_type': 'float32',

    'dilation_rates': 1,
    'do_batch_norm': False,
    'double_gaussian_weights_prior': True,
    'kernel_size': [10, 1, 2],
    'layer_heights': [400, 400, 400, 400, 400, 400, 400, 400, 400],
    'layer_types': ['conv3d', 'conv3d', 'pool3d', 'conv3d', 'conv3d', 'pool3d', 'full', 'full', 'full'],
    'layer_widths': [1, 1, 1, 1, 1, 1, 1, 1, 1],
    'learning_rate': 0.0001,
    'model_save_path': OUTPUT_DIR,
    'n_classification_bins': 2,
    'n_correlated_series': 1,
    'n_epochs': 100,
    'n_eval_passes': 1,
    'n_features_per_series': 271,
    'n_forecasts': 1,
    'n_kernels': 8,
    'n_networks': 1,
    'n_retrain_epochs': 100,
    'n_series': 1,
    'n_train_passes': 1,
    'n_training_samples': 15800,
    'n_training_samples_benchmark': 1000,
    'narrow_prior_std': 0.001,
    'nassets': N_ASSETS,
    'normalise_per_series': False,
    'partial_retrain': False,
    'predict_single_shares': True,
    'random_seed': 0,
    'resume_training': True,
    'retrain_learning_rate': 0.0001,
    'spike_slab_weighting': 0.25,
    'strides': 1,
    'tensorboard_log_path': OUTPUT_DIR,
    'tf_type': 32,
    'train_path': OUTPUT_DIR,
    'use_historical_covariance': True,
    'wide_prior_std': 0.8
}


synthetic_config = {
    "start_date": simulation_start - datetime.timedelta(days=365),
    "end_date": simulation_end + datetime.timedelta(days=10),
    "n_sin_series": 10
}

datasource = SyntheticDataSource(synthetic_config)

oracle_full_config = {
    "scheduling": {
        "prediction_horizon": 24,
        "prediction_frequency": {"frequency_type": "DAILY", "days_offset": 0, "minutes_offset": 60},
        "prediction_delta": 201,
        "training_frequency": {"frequency_type": "WEEKLY", "days_offset": 0, "minutes_offset": 60},
        "training_delta": 201, },
    "oracle": oracle_config
}

oracle_configuration = OracleConfiguration(oracle_full_config)

oracle = CrocubotOracle(oracle_configuration)

scheduler = Scheduler(simulation_start, simulation_end, exchange, oracle.prediction_frequency,
                      oracle.training_frequency, oracle.prediction_horizon)

controller_configuration = ControllerConfiguration({
    'start_date': simulation_start.strftime('%Y-%m-%d'),
    'end_date': simulation_end.strftime('%Y-%m-%d')
})

oracle_performance = OraclePerformance(
    os.path.join(OUTPUT_DIR), 'oracle'
)


controller = Controller(
    configuration=controller_configuration,
    oracle=oracle,
    scheduler=scheduler,
    datasource=datasource,
    performance=oracle_performance
)

controller.run()
