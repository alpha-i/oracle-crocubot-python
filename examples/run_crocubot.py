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
simulation_start = datetime.datetime(2009, 1, 5, tzinfo=pytz.utc)
simulation_end = datetime.datetime(2009, 1, 31, tzinfo=pytz.utc)

oracle_yml = """
data_transformation:
    feature_config_list:
      - is_target: true
        name: close
        normalization: standard
        transformation:
            name: log-return
        local: false
    exchange_name: "{}"
    features_ndays: 10
    features_resample_minutes: 15
    fill_limit: 5
    predict_the_market_close: true

train_path: {}
tensorboard_log_path: {}
covariance_method: Ledoit
covariance_ndays: 100
model_save_path: {}
d_type: float32
tf_type: 32
random_seed: 0
predict_single_shares: True
n_epochs: 200
n_retrain_epochs: 5
n_training_samples: 15800
learning_rate: 0.001
batch_size: 200
cost_type: bayes
n_train_passes: 1
n_eval_passes: 8
resume_training: True
n_series: 1
n_features_per_series: 271
n_forecasts: 1
n_classification_bins: 4
layer_heights: [10, 10, 10, 10]

layer_widths: [1, 1, 1, 1]

layer_types: [full, full, full, full]

activation_functions: [relu, relu, relu, linear]

INITIAL_ALPHA: 0.05
INITIAL_WEIGHT_UNCERTAINTY: 0.02
INITIAL_BIAS_UNCERTAINTY: 0.02
INITIAL_WEIGHT_DISPLACEMENT: 0.1
INITIAL_BIAS_DISPLACEMENT: 0.1
USE_PERFECT_NOISE: False
double_gaussian_weights_prior: True
wide_prior_std: 1.0
narrow_prior_std: 0.001
spike_slab_weighting: 0.25
n_training_samples_benchmark: 1000
n_assets : 10
classify_per_series : False 
normalise_per_series : True
use_historical_covariance : True
n_correlated_series : 1

universe:
    method: "liquidity"
    nassets: 10
    ndays_window: 60 
    update_frequency: "weekly"
    avg_function: "median"
    dropna: False
""".format(exchange, OUTPUT_DIR, OUTPUT_DIR, OUTPUT_DIR)


# synthetic_config = {
#     "start_date": simulation_start - datetime.timedelta(days=365),
#     "end_date": simulation_end + datetime.timedelta(days=10),
#     "n_sin_series": 10
# }

synthetic_config = {
    "start_date": datetime.datetime(2006, 12, 31),
    "end_date": datetime.datetime(2011, 12, 31),
    "n_sin_series": 10
}

datasource = SyntheticDataSource(synthetic_config)

oracle_full_config = {
    "scheduling": {
        "prediction_horizon": 24,
        "prediction_frequency": {"frequency_type": "DAILY", "days_offset": 0, "minutes_offset": 75},
        "prediction_delta": 100,
        "training_frequency": {"frequency_type": "DAILY", "days_offset": 0, "minutes_offset": 60},
        "training_delta": 200, },
    "oracle": yaml.load(oracle_yml)
}

oracle_configuration = OracleConfiguration(oracle_full_config)

oracle = CrocubotOracle(oracle_configuration)

scheduler = Scheduler(simulation_start, simulation_end, exchange, oracle.prediction_frequency, oracle.training_frequency)

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
