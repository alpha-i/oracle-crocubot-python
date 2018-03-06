import datetime
import os
import tempfile

import pytz
import yaml
from alphai_delphi.controller import ControllerConfiguration, Controller
from alphai_delphi.data_source.synthetic_data_source import SyntheticDataSource
from alphai_delphi.oracle.oracle_configuration import OracleConfiguration
from alphai_delphi.performance.performance import OraclePerformance
from alphai_delphi.scheduler import Scheduler

ORACLE_CONFIGURATION_YML = """
data_transformation:
    feature_config_list:
      - is_target: true
        name: close
        normalization: standard
        transformation:
            name: log-return
        local: false
    exchange_name: "NYSE"
    features_ndays: 10
    features_resample_minutes: 15
    fill_limit: 5
    predict_the_market_close: true

train_path: {0}
tensorboard_log_path: {0}
covariance_method: Ledoit
covariance_ndays: 100
model_save_path: {0}
d_type: float32
tf_type: 32
random_seed: 0
predict_single_shares: True
n_epochs: 200
n_retrain_epochs: 1
n_training_samples: 15800
learning_rate: 0.001
batch_size: 200
cost_type: bayes
n_train_passes: 1
n_eval_passes: 1
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
nassets : 10
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
"""

SYNTHETIC_DATASOURCE = SyntheticDataSource({
    "start_date": datetime.datetime(2006, 12, 31),
    "end_date": datetime.datetime(2011, 12, 31),
    "n_sin_series": 10
})


class BaseIntegration:
    ORACLE_CLASS = None
    ORACLE_CONFIGURATION_YML = ORACLE_CONFIGURATION_YML
    OUTPUT_DIR = tempfile.gettempdir()
    SIMULATION_START = datetime.datetime(2009, 1, 5, tzinfo=pytz.utc)
    SIMULATION_END = datetime.datetime(2009, 1, 31, tzinfo=pytz.utc)
    CALENDAR_NAME = 'NYSE'
    DATASOURCE = SYNTHETIC_DATASOURCE

    def setUp(self):
        self.oracle_full_config = {
            "scheduling": {
                "prediction_horizon": 24,
                "prediction_frequency": {"frequency_type": "WEEKLY", "days_offset": 0, "minutes_offset": 75},
                "prediction_delta": 100,
                "training_frequency": {"frequency_type": "WEEKLY", "days_offset": 0, "minutes_offset": 60},
                "training_delta": 200, },
            "oracle": yaml.load(self.ORACLE_CONFIGURATION_YML.format(self.OUTPUT_DIR))
        }
        self.controller_configuration = ControllerConfiguration({
            'start_date': '2009-01-05',
            'end_date': '2009-01-31'
        })

        self.oracle = self.ORACLE_CLASS(
            OracleConfiguration(self.oracle_full_config)
        )

        self.scheduler = Scheduler(
            self.SIMULATION_START,
            self.SIMULATION_END,
            self.CALENDAR_NAME,
            self.oracle.prediction_frequency,
            self.oracle.training_frequency,
            self.oracle.prediction_horizon
        )

        self.oracle_performance = OraclePerformance(
            os.path.join(self.OUTPUT_DIR), 'oracle'
        )

        self.controller = Controller(
            configuration=self.controller_configuration,
            oracle=self.oracle,
            scheduler=self.scheduler,
            datasource=self.DATASOURCE,
            performance=self.oracle_performance
        )

    def test_run(self):
        self.controller.run()
