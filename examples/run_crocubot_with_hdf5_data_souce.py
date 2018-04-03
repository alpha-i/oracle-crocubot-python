import os
import datetime
import pytz
import yaml

import logging

os.environ['MKL_THREADING_LAYER'] = 'GNU'

from alphai_delphi.controller import ControllerConfiguration, Controller
from alphai_delphi.data_source.hdf5_data_source import StocksHDF5DataSource
from alphai_delphi.oracle.oracle_configuration import OracleConfiguration
from alphai_delphi.performance.performance import OraclePerformance
from alphai_delphi.scheduler import Scheduler

from alphai_crocubot_oracle.oracle import CrocubotOracle

logging.basicConfig(level=logging.DEBUG)

OUTPUT_DIR = './result'

exchange = "NYSE"
simulation_start = datetime.datetime(2009, 1, 5, tzinfo=pytz.utc)
simulation_end = datetime.datetime(2009, 1, 31, tzinfo=pytz.utc)

data_source_config = {
   "filename": "/home/sbalan/Documents/QQ_data/Q_20061231_20111231_SP500_adjusted_1m_float32.hdf5",
   "exchange": "NYSE",
   "data_timezone": "America/New_York",
   "start": datetime.datetime(2006, 12, 31),
   "end": datetime.datetime(2011, 12, 31)
}
datasource = StocksHDF5DataSource(data_source_config)

qw_config_file_name = "/home/sbalan/github/configs_for_testing_qw_delphi/crocubot_config_qw.yml"
with open(qw_config_file_name, 'r') as qw_config_file:
    qw_config = yaml.load(qw_config_file)

oracle_config = qw_config['quant_workflow']['oracle']['oracle_arguments']
oracle_config['universe'] = qw_config['quant_workflow']['universe']
oracle_config['universe']['dropna'] = False

oracle_config['train_path'] = OUTPUT_DIR
oracle_config['tensorboard_log_path'] = OUTPUT_DIR
oracle_config['model_save_path'] = OUTPUT_DIR


oracle_full_config = {
    "scheduling": {
        "prediction_horizon": 24,
        "prediction_frequency": {"frequency_type": "DAILY", "days_offset": 0, "minutes_offset": 75},
        "prediction_delta": 250,
        "training_frequency": {"frequency_type": "WEEKLY", "days_offset": 0, "minutes_offset": 60},
        "training_delta": 300, },
    "oracle": oracle_config
}

oracle_configuration = OracleConfiguration(oracle_full_config)

oracle = CrocubotOracle(oracle_configuration)

scheduler = Scheduler(simulation_start, simulation_end, exchange, oracle.prediction_frequency,
                      oracle.training_frequency)

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
