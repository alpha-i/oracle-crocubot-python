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

EXCHANGE_NAME = "JPX"
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

scheduling = %%SCHEDULE_CONFIG%%

###########################################
#      ALPHA-I ORACLE CONFIGURATION       #
###########################################

N_ASSETS = 400

oracle_config = %%ORACLE_CONFIG%%

logging.info("Loading datasource ...")
datasource = AlcovaDataSource(datasource_configuration)
logging.info("Datasource Loaded!")

oracle = CrocubotOracle(OracleConfiguration({"scheduling": scheduling, "oracle": oracle_config}))

scheduler = Scheduler(simulation_start, simulation_end, EXCHANGE_NAME,
                      oracle.prediction_frequency, oracle.training_frequency, oracle.prediction_horizon)

oracle_performance = OraclePerformance(os.path.join(RESULT_DIRECTORY), run_name)

controller = Controller(
    configuration=ControllerConfiguration(controller_config),
    oracle=oracle,
    scheduler=scheduler,
    datasource=datasource,
    performance=oracle_performance
)
