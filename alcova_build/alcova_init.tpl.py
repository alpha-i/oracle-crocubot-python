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

logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

EXCHANGE_NAME = "JPX"

run_name = 'alcova'


class SimulationEnvironment:

    def __init__(self, alcova_config, base_dir):

        self.alcova_config = yaml.load(alcova_config)
        self.base_dir = base_dir

        self.datasource = None
        self.alcova_config = None

        self.oracle = None
        self.oracle_performance = None

        self.scheduler = None
        self.controller = None

        RUNTIME_DIR_PATH = os.path.join(self.base_dir, 'runtime')

        self.oracle_config = %%ORACLE_CONFIG%%

        self.scheduling = %%SCHEDULE_CONFIG%%

    def run(self):
        """
        Full initialization
        :return:
        """
        self.initialize_datasource()
        self.initialize_oracle()
        self.initialize_simulation()
        self.controller.run()

    def initialize_simulation(self):
        """
        Initialize Scheduler, Controller and Performance

        :return:
        """

        if not self.oracle:
            raise RuntimeError('Must initialize oracle first')

        if not self.datasource:
            raise RuntimeError('Must initialize datasource before the simulation')

        simulation_start = pytz.utc.localize(datetime.strptime(self.alcova_config['simulation_start'], DATE_FORMAT))
        simulation_end = pytz.utc.localize(datetime.strptime(self.alcova_config['simulation_end'], DATE_FORMAT))

        self.scheduler = Scheduler(simulation_start,
                                   simulation_end,
                                   EXCHANGE_NAME,
                                   self.oracle.prediction_frequency,
                                   self.oracle.training_frequency,
                                   self.oracle.prediction_horizon)

        self.oracle_performance = OraclePerformance(os.path.join(self.base_dir, 'result'), run_name)

        controller_configuration = ControllerConfiguration(
            {'start_date': self.alcova_config['simulation_start'], 'end_date': self.alcova_config['simulation_end']}
        )

        self.controller = Controller(
            configuration=controller_configuration,
            oracle=self.oracle,
            scheduler=self.scheduler,
            datasource=self.datasource,
            performance=self.oracle_performance
        )

    def initialize_oracle(self):
        """
        Build the oracle
        :return:
        """
        self.oracle = CrocubotOracle(OracleConfiguration({"scheduling": self.scheduling,
                                                          "oracle": self.oracle_config}))

    def initialize_datasource(self):
        """
        Build the datasource
        :return:
        """
        datasource_configuration = {
            'feature_mapping': {
                'close': self.alcova_config['data_files']['close'],
                'volume': self.alcova_config['data_files']['volume']
            },
            'adjustments_file': self.alcova_config['data_files']['adjustments']
        }
        logging.info("Loading datasource ...")
        self.datasource = AlcovaDataSource(datasource_configuration)
        logging.info("Datasource Loaded!")
