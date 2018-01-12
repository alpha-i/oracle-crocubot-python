from unittest import TestCase
from tests.helpers import (create_fixtures, destroy_fixtures, read_hdf5_into_dict_of_data_frames, FIXTURE_DATA_FULLPATH)
import pandas as pd
from alphai_crocubot_oracle.covariance import estimate_covariance, DEFAULT_NUM_REALISATIONS_MULTIPLICATION_FACTOR, \
    returns_minutes_after_market_open_data_frame
from alphai_feature_generation.transformation import FinancialDataTransformation

from sklearn.covariance import GraphLassoCV
import numpy as np


class TestCrocubot(TestCase):
    def setUp(self):
        create_fixtures()

    def tearDown(self):
        destroy_fixtures()

    def _prepare_data_for_test(self):
        start_date = '20140102'  # these are values for the resources/sample_hdf5.h5
        end_date = '20140228'
        symbols = ['AAPL', 'INTC', 'MSFT']
        exchange_name = 'NYSE'
        fill_limit = 10
        resample_rule = '15T'

        historical_universes = pd.DataFrame(columns=['start_date', 'end_date', 'assets'])
        historical_universes.loc[0] = [pd.Timestamp(start_date), pd.Timestamp(end_date), symbols]
        data = read_hdf5_into_dict_of_data_frames(start_date, end_date, symbols, FIXTURE_DATA_FULLPATH, exchange_name,
                                                  fill_limit, resample_rule)
        return historical_universes, data

    def test_estimate_covariance(self):
        configuration = {
            'feature_config_list': [
                {
                    'name': 'close',
                    'normalization': 'standard',
                    'transformation': {'name': 'log-return'},
                    'is_target': True,
                    'local': False,
                },
            ],
            'fill_limit': 0,
            'exchange_name': 'NYSE',
            'features_ndays': 9,
            'features_resample_minutes': 15,
            'features_start_market_minute': 60,
            'prediction_frequency_ndays': 1,
            'prediction_market_minute': 60,
            'target_delta_ndays': 1,
            'target_market_minute': 60,
            'n_classification_bins': 12,
            'nassets': 3,
            'classify_per_series': False,
            'normalise_per_series': False
        }

        data_transformation = FinancialDataTransformation(configuration)
        universe, data = self._prepare_data_for_test()
        estimation_method = "Ledoit"
        exchange_calendar = data_transformation.exchange_calendar
        ndays = data_transformation.features_ndays  # FIXME this is the only value that works now.
        forecast_interval = data_transformation.target_delta_ndays
        target_market_minute = data_transformation.target_market_minute
        covariance_matrix = estimate_covariance(data, ndays, target_market_minute, estimation_method, exchange_calendar,
                                                forecast_interval)

        ret_data = returns_minutes_after_market_open_data_frame(data['close'], exchange_calendar, target_market_minute)
        print(ret_data.shape)
        nd = ret_data.shape[1]
        sampling_days = nd * DEFAULT_NUM_REALISATIONS_MULTIPLICATION_FACTOR
        data_points = ret_data.values[-sampling_days:, :]
        glass_model = GraphLassoCV()
        glass_model.fit(data_points)
        cov_mat = glass_model.covariance_
        self.assertTrue(np.allclose(covariance_matrix.diagonal(), cov_mat.diagonal()))
