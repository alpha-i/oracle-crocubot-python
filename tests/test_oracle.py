import os
from datetime import datetime, timedelta
from unittest import TestCase

import pandas as pd

from alphai_crocubot_oracle import DATETIME_FORMAT_COMPACT
from alphai_crocubot_oracle.oracle import TRAIN_FILE_NAME_TEMPLATE

from tests.helpers import (
    load_default_config,
    FIXTURE_DESTINATION_DIR,
    FIXTURE_DATA_FULLPATH,
    create_fixtures,
    destroy_fixtures,
    read_hdf5_into_dict_of_data_frames,
    DummyCrocubotOracle,
)


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

        historical_universes = pd.DataFrame(columns=['start_date', 'end_date', 'assets'])
        historical_universes.loc[0] = [
            pd.Timestamp(start_date),
            pd.Timestamp(end_date),
            symbols,
        ]
        data = read_hdf5_into_dict_of_data_frames(start_date,
                                                  end_date,
                                                  symbols,
                                                  FIXTURE_DATA_FULLPATH,
                                                  exchange_name,
                                                  fill_limit,
                                                  resample_rule
                                                  )
        return historical_universes, data

    def test_crocubot_train_and_predict(self):
        historical_universes, data = self._prepare_data_for_test()

        configuration = load_default_config()
        configuration['n_correlated_series'] = 1
        model = DummyCrocubotOracle(configuration)

        train_time = datetime(2017, 6, 7, 9) + timedelta(minutes=60)
        prediction_time = train_time + timedelta(minutes=1)

        model.train(historical_universes, data, train_time)

        _, predict_data = self._prepare_data_for_test()
        model.predict(predict_data, prediction_time)

    def test_crocubot_train_and_save_file(self):
        train_time = datetime(2017, 6, 7, 9) + timedelta(minutes=60)
        train_filename = TRAIN_FILE_NAME_TEMPLATE.format(train_time.strftime(DATETIME_FORMAT_COMPACT))

        expected_train_path = os.path.join(FIXTURE_DESTINATION_DIR, train_filename)

        historical_universes, data = self._prepare_data_for_test()

        configuration = load_default_config()
        model = DummyCrocubotOracle(configuration)

        model.train(historical_universes, data, train_time)

        tf_suffix = '.index'  # TF adds stuff to the end of its save files
        full_tensorflow_path = expected_train_path + tf_suffix
        self.assertTrue(os.path.exists(full_tensorflow_path))

    def test_crocubot_predict_without_train_file(self):
        configuration = load_default_config()
        model = DummyCrocubotOracle(configuration)

        execution_time = datetime(2017, 6, 7, 9) + timedelta(minutes=60)

        _, predict_data = self._prepare_data_for_test()
        self.assertRaises(
            ValueError,
            model.predict,
            predict_data,
            execution_time
        )
