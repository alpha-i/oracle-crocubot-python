import os
from datetime import datetime, timedelta
from unittest import TestCase

from alphai_delphi.oracle.oracle_configuration import OracleConfiguration

from alphai_crocubot_oracle import DATETIME_FORMAT_COMPACT
from alphai_crocubot_oracle.oracle import TRAIN_FILE_NAME_TEMPLATE

from tests.helpers import (
    default_oracle_config,
    FIXTURE_DESTINATION_DIR,
    FIXTURE_DATA_DESTINATION_FILE,
    create_fixtures,
    destroy_fixtures,
    read_hdf5_into_dict_of_data_frames,
    DummyCrocubotOracle,
    default_scheduling_config)


class TestCrocubot(TestCase):
    def setUp(self):
        create_fixtures()

    def tearDown(self):
        destroy_fixtures()

    @staticmethod
    def _prepare_data_for_test():
        start_date = '20140102'  # these are values for the resources/sample_hdf5.h5
        end_date = '20140228'
        symbols = ['AAPL', 'INTC', 'MSFT']
        exchange_name = 'NYSE'
        fill_limit = 10
        resample_rule = '15T'

        data = read_hdf5_into_dict_of_data_frames(start_date,
                                                  end_date,
                                                  symbols,
                                                  FIXTURE_DATA_DESTINATION_FILE,
                                                  exchange_name,
                                                  fill_limit,
                                                  resample_rule
                                                  )
        return data

    def test_crocubot_train_and_predict(self):
        data = self._prepare_data_for_test()

        oracle_configuration = default_oracle_config()
        oracle_configuration['n_correlated_series'] = 1

        configuration = OracleConfiguration({
            'scheduling': default_scheduling_config(),
            'oracle': oracle_configuration
        })

        model = DummyCrocubotOracle(configuration)

        current_timestamp = datetime(2014, 1, 20, 9) + timedelta(minutes=60)
        target_timestamp = current_timestamp + model.prediction_horizon

        model.train(data, current_timestamp)

        predict_data = self._prepare_data_for_test()
        model.predict(predict_data, current_timestamp, target_timestamp)

    def test_crocubot_train_and_save_file(self):
        train_time = datetime(2014, 1, 20, 9) + timedelta(minutes=60)
        train_filename = TRAIN_FILE_NAME_TEMPLATE.format(train_time.strftime(DATETIME_FORMAT_COMPACT))

        expected_train_path = os.path.join(FIXTURE_DESTINATION_DIR, train_filename)

        data = self._prepare_data_for_test()
        configuration = OracleConfiguration({
            'scheduling': default_scheduling_config(),
            'oracle': default_oracle_config()
        })

        model = DummyCrocubotOracle(configuration)

        current_time = datetime(2014, 1, 20, 9) + timedelta(minutes=60)

        model.train(data, current_time)

        tf_suffix = '.index'  # TF adds stuff to the end of its save files
        full_tensorflow_path = expected_train_path + tf_suffix
        self.assertTrue(os.path.exists(full_tensorflow_path))

    def test_crocubot_predict_without_train_file(self):

        configuration = OracleConfiguration({
            'scheduling': default_scheduling_config(),
            'oracle': default_oracle_config()
        })

        model = DummyCrocubotOracle(configuration)

        current_timestamp = datetime(2014, 1, 20, 9) + timedelta(minutes=60)
        target_timestamp = current_timestamp + model.prediction_horizon

        predict_data = self._prepare_data_for_test()
        self.assertRaises(
            ValueError,
            model.predict,
            predict_data,
            current_timestamp,
            target_timestamp
        )
