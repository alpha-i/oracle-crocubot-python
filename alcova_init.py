from datetime import datetime
import logging
import os
import yaml
import pytz

os.environ['MKL_THREADING_LAYER'] = 'GNU'

from alphai_delphi.controller import ControllerConfiguration, Controller
from alphai_alcova_datasource.data_source import AlcovaDataSource
from alphai_delphi.oracle.oracle_configuration import OracleConfiguration
from alphai_delphi.performance.performance import OraclePerformance
from alphai_delphi.scheduler import Scheduler

from alphai_crocubot_oracle.oracle import CrocubotOracle

LOG_FORMAT = '%(asctime)s - %(levelname)s [%(name)s:%(module)s:%(lineno)d]: %(message)s'
DATE_FORMAT = '%Y-%m-%d'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

with open(os.path.join(BASE_DIR, 'alcova.yml')) as alcova_config_file:
    alcova_config = yaml.load(alcova_config_file)

exchange_name = "JPX"
simulation_start = pytz.utc.localize(datetime.strptime(alcova_config['simulation_start'], DATE_FORMAT))
simulation_end = pytz.utc.localize(datetime.strptime(alcova_config['simulation_end'], DATE_FORMAT))
run_name = 'alcova'

RUNTIME_DIR_PATH = os.path.join(BASE_DIR, 'runtime')
RESULT_DIRECTORY = os.path.join(BASE_DIR, 'result')

datasource_configuration = {
    'feature_mapping': {
        'close': alcova_config['data_files']['close'],
        'volume': alcova_config['data_files']['volume']
    },
    'adjustments_file': alcova_config['data_files']['adjustments']
}


###########################################
#         RUNTIME CONFIGURATIONS          #
###########################################

controller_config = {
    'start_date': alcova_config['simulation_start'],
    'end_date': alcova_config['simulation_end']
}

scheduling = {
    "prediction_horizon": 24,
    "prediction_frequency": {"frequency_type": "DAILY", "days_offset": 0, "minutes_offset": 60},
    "prediction_delta": 100,
    "training_frequency": {"frequency_type": "WEEKLY", "days_offset": 0, "minutes_offset": 60},
    "training_delta": 200,
}

###########################################
#      ALPHA-I ORACLE CONFIGURATION       #
###########################################

N_ASSETS = 400

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
        'exchange_name': exchange_name,
        'features_ndays': 100,
        'features_resample_minutes': 10,
        'fill_limit': 0,
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
    'model_save_path': RUNTIME_DIR_PATH,
    'n_classification_bins': 2,
    'n_correlated_series': 1,
    'n_epochs': 1000,
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
    'tensorboard_log_path': RUNTIME_DIR_PATH,
    'tf_type': 32,
    'train_path': RUNTIME_DIR_PATH,
    'use_historical_covariance': True,
    'wide_prior_std': 0.8
}


datasource = AlcovaDataSource(datasource_configuration)
oracle = CrocubotOracle(OracleConfiguration({"scheduling": scheduling, "oracle": oracle_config}))

scheduler = Scheduler(simulation_start, simulation_end, exchange_name,
                      oracle.prediction_frequency, oracle.training_frequency, oracle.prediction_horizon)

oracle_performance = OraclePerformance(os.path.join(RESULT_DIRECTORY), run_name)

controller = Controller(
    configuration=ControllerConfiguration(controller_config),
    oracle=oracle,
    scheduler=scheduler,
    datasource=datasource,
    performance=oracle_performance
)
