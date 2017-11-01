from datetime import timedelta
from unittest import TestCase

import numpy as np
import pandas as pd
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal

from alphai_crocubot_oracle.data.classifier import BinDistribution
from alphai_crocubot_oracle.data.feature import (
    FinancialFeature,
    financial_features_factory,
    single_financial_feature_factory,
    get_feature_names,
    get_feature_max_ndays
)
from tests.data.helpers import (
    sample_market_calendar,
    sample_hourly_ohlcv_data_dict,
    sample_fin_feature_factory_list,
    sample_fin_feature_list,
    TEST_ARRAY
)

SAMPLE_TRAIN_LABELS = np.stack((TEST_ARRAY, TEST_ARRAY, TEST_ARRAY, TEST_ARRAY, TEST_ARRAY))
SAMPLE_PREDICT_LABELS = SAMPLE_TRAIN_LABELS[:, int(0.5 * SAMPLE_TRAIN_LABELS.shape[1])]

SAMPLE_TRAIN_LABELS = {'open': SAMPLE_TRAIN_LABELS}
SAMPLE_PREDICT_LABELS = {'open': SAMPLE_PREDICT_LABELS}

ASSERT_NDECIMALS = 5


class TestFinancialFeature(TestCase):
    def setUp(self):
        self.feature_1 = FinancialFeature(
            name='open',
            transformation={'name': 'value'},
            normalization=None,
            nbins=5,
            ndays=2,
            resample_minutes=60,
            start_market_minute=30,
            is_target=True,
            exchange_calendar=sample_market_calendar,
        )
        self.feature_2 = FinancialFeature(
            name='close',
            transformation={'name': 'log-return'},
            normalization=None,
            nbins=10,
            ndays=5,
            resample_minutes=60,
            start_market_minute=90,
            is_target=True,
            exchange_calendar=sample_market_calendar,
        )
        self.feature_3 = FinancialFeature(
            name='high',
            transformation={'name': 'log-return'},
            normalization='standard',
            nbins=None,
            ndays=10,
            resample_minutes=60,
            start_market_minute=150,
            is_target=True,
            exchange_calendar=sample_market_calendar,
        )
        self.feature_4 = FinancialFeature(
            name='high',
            transformation={'name': 'stochastic_k'},
            normalization=None,
            nbins=10,
            ndays=10,
            resample_minutes=60,
            start_market_minute=150,
            is_target=True,
            exchange_calendar=sample_market_calendar,
        )
        self.feature_5 = FinancialFeature(
            name='high',
            transformation={'name': 'ewma', 'halflife': 20},
            normalization=None,
            nbins=10,
            ndays=10,
            resample_minutes=60,
            start_market_minute=150,
            is_target=True,
            exchange_calendar=sample_market_calendar,
        )
        self.feature_6 = FinancialFeature(
            name='high',
            transformation={'name': 'KER', 'lag': 20},
            normalization=None,
            nbins=10,
            ndays=10,
            resample_minutes=60,
            start_market_minute=150,
            is_target=True,
            exchange_calendar=sample_market_calendar,
        )
        self.feature_7 = FinancialFeature(
            name='high',
            transformation={'name': 'log-return'},
            normalization='standard',
            nbins=10,
            ndays=10,
            resample_minutes=60,
            start_market_minute=150,
            is_target=True,
            exchange_calendar=sample_market_calendar,
        )

    def test_process_prediction_data_x_1(self):
        data_frame_x = sample_hourly_ohlcv_data_dict[self.feature_1.name]
        processed_prediction_data_x = self.feature_1.process_prediction_data_x(data_frame_x)
        assert processed_prediction_data_x.equals(data_frame_x)

    def test_process_prediction_data_x_2(self):
        data_frame_x = sample_hourly_ohlcv_data_dict[self.feature_2.name]
        processed_prediction_data_x = self.feature_2.process_prediction_data_x(data_frame_x)
        expected_log_returns = np.log(data_frame_x.pct_change() + 1). \
            replace([np.inf, -np.inf], np.nan).dropna()
        assert_almost_equal(processed_prediction_data_x, expected_log_returns.values, ASSERT_NDECIMALS)

    def test_process_prediction_data_x_3(self):
        data_frame_x = sample_hourly_ohlcv_data_dict[self.feature_3.name]
        processed_prediction_data_x = self.feature_3.process_prediction_data_x(data_frame_x)
        expected_normalized_log_returns = \
            (np.log(data_frame_x.pct_change() + 1).replace([np.inf, -np.inf], np.nan).dropna()).values
        assert_almost_equal(processed_prediction_data_x, expected_normalized_log_returns, ASSERT_NDECIMALS)

    def test_process_prediction_data_x_4(self):
        data_frame_x = sample_hourly_ohlcv_data_dict[self.feature_4.name]
        processed_prediction_data_x = self.feature_4.process_prediction_data_x(data_frame_x)

        columns = data_frame_x.columns

        expected_result = ((data_frame_x.iloc[-1] - data_frame_x.min()) /
                           (data_frame_x.max() - data_frame_x.min())) * 100.

        expected_result = np.expand_dims(expected_result, axis=0)
        expected_result = pd.DataFrame(expected_result, columns=columns)

        assert_almost_equal(processed_prediction_data_x.values, expected_result.values, ASSERT_NDECIMALS)

    def test_process_prediction_data_x_5(self):
        data_frame_x = sample_hourly_ohlcv_data_dict[self.feature_5.name]
        processed_prediction_data_x = self.feature_5.process_prediction_data_x(data_frame_x)

        expected_result = data_frame_x.ewm(halflife=self.feature_5.transformation['halflife']).mean()
        assert_almost_equal(processed_prediction_data_x.values, expected_result.values, ASSERT_NDECIMALS)

    def test_process_prediction_data_x_6(self):
        data_frame_x = sample_hourly_ohlcv_data_dict[self.feature_6.name]
        processed_prediction_data_x = self.feature_6.process_prediction_data_x(data_frame_x)

        direction = data_frame_x.diff(self.feature_6.transformation['lag']).abs()
        volatility = data_frame_x.diff().abs().rolling(window=self.feature_6.transformation['lag']).sum()

        direction.dropna(axis=0, inplace=True)
        volatility.dropna(axis=0, inplace=True)

        expected_result = direction / volatility
        expected_result.dropna(axis=0, inplace=True)

        assert_almost_equal(processed_prediction_data_x.values, expected_result.values, ASSERT_NDECIMALS)

    def test_process_prediction_data_y_1(self):
        data_frame = sample_hourly_ohlcv_data_dict[self.feature_1.name]
        data_frame_x = data_frame.iloc[:-1]
        prediction_reference_data = data_frame_x.iloc[-1]
        data_frame_y = data_frame.iloc[-1]
        self.feature_1.process_prediction_data_x(data_frame_x)
        processed_prediction_data_y = \
            self.feature_1.process_prediction_data_y(data_frame_y, prediction_reference_data)
        assert processed_prediction_data_y.equals(data_frame_y)

    def test_process_prediction_data_y_2(self):

        data_frame = sample_hourly_ohlcv_data_dict[self.feature_2.name]
        data_frame_x = data_frame.iloc[:-1]
        prediction_reference_data = data_frame_x.iloc[-1]
        data_frame_y = data_frame.iloc[-1]
        self.feature_2.process_prediction_data_x(data_frame_x)
        processed_prediction_data_y = \
            self.feature_2.process_prediction_data_y(data_frame_y, prediction_reference_data)
        expected_log_returns = np.log(data_frame_y / prediction_reference_data)
        assert_almost_equal(processed_prediction_data_y, expected_log_returns.values, ASSERT_NDECIMALS)

    def test_process_prediction_data_y_3(self):
        data_frame = sample_hourly_ohlcv_data_dict[self.feature_3.name]
        data_frame_x = data_frame.iloc[:-1]
        prediction_reference_data = data_frame_x.iloc[-1]
        data_frame_y = data_frame.iloc[-1]
        self.feature_3.process_prediction_data_x(data_frame_x)

        self.assertRaises(NotImplementedError, self.feature_3.process_prediction_data_y,
                          data_frame_y, prediction_reference_data)

    def test_process_prediction_data_y_7(self):
        data_frame = sample_hourly_ohlcv_data_dict[self.feature_7.name]
        data_frame_x = data_frame.iloc[:-1]
        prediction_reference_data = data_frame_x.iloc[-1]
        data_frame_y = data_frame.iloc[-1]
        self.feature_7.process_prediction_data_x(data_frame_x)

        processed_prediction_data_y = \
            self.feature_7.process_prediction_data_y(data_frame_y, prediction_reference_data)

        log_ratio_data = np.log(data_frame_y / prediction_reference_data)
        expected_normalized_log_returns = log_ratio_data.values

        assert_almost_equal(processed_prediction_data_y, expected_normalized_log_returns, ASSERT_NDECIMALS)

    def test_get_start_timestamp_x(self):
        start_date_str = '20150101'
        end_date_str = '20150501'

        market_open_list = sample_market_calendar.schedule(start_date_str, end_date_str).market_open
        prediction_timestamp = market_open_list[20] + timedelta(minutes=15)

        start_timestamp_x_1 = self.feature_1._get_start_timestamp_x(prediction_timestamp)
        expected_start_timestamp_x1 = pd.Timestamp('2015-01-29 15:00:00+0000', tz='UTC')
        assert start_timestamp_x_1 == expected_start_timestamp_x1

        start_timestamp_x_2 = self.feature_2._get_start_timestamp_x(prediction_timestamp)
        expected_start_timestamp_x2 = pd.Timestamp('2015-01-26 16:00:00+0000', tz='UTC')
        assert start_timestamp_x_2 == expected_start_timestamp_x2

        start_timestamp_x_3 = self.feature_3._get_start_timestamp_x(prediction_timestamp)
        expected_start_timestamp_x3 = pd.Timestamp('2015-01-16 17:00:00+0000', tz='UTC')
        assert start_timestamp_x_3 == expected_start_timestamp_x3

    def test_select_prediction_data(self):
        data_frame = sample_hourly_ohlcv_data_dict[self.feature_1.name]
        start_date = data_frame.index[0].date()
        end_date = data_frame.index[-1].date()

        market_open_list = sample_market_calendar.schedule(str(start_date), str(end_date)).market_open
        prediction_timestamp = market_open_list[20] + timedelta(minutes=15)

        selected_prediction_data = \
            self.feature_1._select_prediction_data_x(data_frame, prediction_timestamp)

        start_timestamp_x_1 = self.feature_1._get_start_timestamp_x(prediction_timestamp)
        expected_data_frame = data_frame[(data_frame.index >= start_timestamp_x_1) &
                                         (data_frame.index <= prediction_timestamp)]
        assert selected_prediction_data.equals(expected_data_frame)

    @staticmethod
    def run_get_prediction_data_test(feature, expected_length):
        data_frame = sample_hourly_ohlcv_data_dict[feature.name]
        start_date = data_frame.index[0].date()
        end_date = data_frame.index[-1].date()

        market_open_list = sample_market_calendar.schedule(str(start_date), str(end_date)).market_open
        prediction_timestamp = market_open_list[20] + timedelta(minutes=30)
        target_timestamp = market_open_list[21] + timedelta(minutes=90)

        prediction_data_x, prediction_data_y = \
            feature.get_prediction_data(data_frame, prediction_timestamp, target_timestamp)

        assert isinstance(prediction_data_x, pd.DataFrame) and isinstance(prediction_data_y, pd.Series)
        assert len(prediction_data_x) == expected_length
        assert_array_equal(prediction_data_x.columns, prediction_data_y.index)

    def test_get_prediction_data(self):
        feature_list = [self.feature_1, self.feature_2, self.feature_7]
        expected_length_list = [15, 34, 68]
        for feature, expected_length in zip(feature_list, expected_length_list):
            self.run_get_prediction_data_test(feature, expected_length)

    def test_calculate_bin_distribution(self):
        n_train = 20
        feature_list = [self.feature_1, self.feature_2, self.feature_3]
        train_y = np.linspace(0, 1, n_train)
        for feature in feature_list:
            if feature.nbins:
                feature.calculate_bin_distribution(train_y)
                assert isinstance(feature.bin_distribution, BinDistribution)
            else:
                with pytest.raises(AssertionError):
                    feature.calculate_bin_distribution(train_y)

    def test_classify_train_data_y(self):
        feature_list = [self.feature_1, self.feature_2, self.feature_3]
        for feature in feature_list:
            classified_train_y = feature.classify_train_data_y(SAMPLE_TRAIN_LABELS[list(SAMPLE_TRAIN_LABELS.keys())[0]])
            if feature.nbins:
                assert isinstance(feature.bin_distribution, BinDistribution)
                assert classified_train_y.shape \
                    == SAMPLE_TRAIN_LABELS[list(SAMPLE_TRAIN_LABELS.keys())[0]].shape + (feature.nbins,)
                assert_almost_equal(
                    classified_train_y.sum(axis=1),
                    (SAMPLE_TRAIN_LABELS[list(SAMPLE_TRAIN_LABELS.keys())[0]].shape[1] /
                     feature.nbins * np.ones(shape=(len(SAMPLE_TRAIN_LABELS[list(SAMPLE_TRAIN_LABELS.keys())[0]]),
                                                    feature.nbins))),
                    ASSERT_NDECIMALS)
            else:
                assert_almost_equal(classified_train_y,
                                    SAMPLE_TRAIN_LABELS[list(SAMPLE_TRAIN_LABELS.keys())[0]],
                                    ASSERT_NDECIMALS)

    def test_declassify_single_predict_y(self):
        feature_list = [self.feature_1, self.feature_2, self.feature_3]
        for feature in feature_list:
            if feature.nbins:
                predict_y = np.zeros_like(SAMPLE_PREDICT_LABELS[list(SAMPLE_PREDICT_LABELS.keys())[0]])
                predict_y[0] = 1
            else:
                predict_y = SAMPLE_PREDICT_LABELS
            with pytest.raises(NotImplementedError):
                feature.declassify_single_predict_y(predict_y)

    def test_declassify_multi_predict_y(self):
        n_passes = 10
        n_train = 20
        n_series = 5
        feature_list = [self.feature_1, self.feature_2, self.feature_3]
        train_y = {'open': np.expand_dims(np.linspace(0, 1, n_train), 0)}
        expected_means = n_series * [0.5]
        expected_variances_list = [
            n_series * [0.07666667],
            n_series * [0.08166667],
            n_series * [0.10185185],
        ]
        for idx, feature in enumerate(feature_list):
            feature.classify_train_data_y(train_y[list(train_y.keys())[0]])
            if feature.nbins:
                predict_y = np.zeros((n_passes, n_series, feature.nbins))
                for i in range(n_passes):
                    for j in range(n_series):
                        predict_y[i, j, i % feature.nbins] = 1
            else:
                predict_y = np.stack(n_series * [np.linspace(0, 1, n_passes)]).transpose()

            means, variances = feature.declassify_multi_predict_y(predict_y)

            assert_almost_equal(means, expected_means, ASSERT_NDECIMALS)
            assert_almost_equal(variances, expected_variances_list[idx], ASSERT_NDECIMALS)

    def test_inverse_transform_multi_predict_y(self):
        n_passes = 10
        n_train = 20
        n_series = 5
        feature_list = [self.feature_1, self.feature_2, self.feature_3]
        train_y = {'open': np.expand_dims(np.linspace(0, 1, n_train), 0)}
        expected_means_list = [
            n_series * [0.5],
            n_series * [0.5],
            n_series * [0.5],
        ]
        expected_variances_list = [
            n_series * [0.07666667],
            n_series * [0.08166667],
            n_series * [0.10185185],
        ]
        for idx, feature in enumerate(feature_list):

            feature.classify_train_data_y(train_y[list(train_y.keys())[0]])

            data_frame_x = sample_hourly_ohlcv_data_dict[feature.name]
            feature.process_prediction_data_x(data_frame_x)

            if feature.nbins:
                predict_y = np.zeros((n_passes, n_series, feature.nbins))
                for i in range(n_passes):
                    for j in range(n_series):
                        predict_y[i, j, i % feature.nbins] = 1
            else:
                predict_y = np.stack(n_series * [np.linspace(0, 1, n_passes)]).transpose()

            means, cov_matrix = feature.inverse_transform_multi_predict_y(predict_y)

            assert_almost_equal(means, expected_means_list[idx], ASSERT_NDECIMALS)
            assert_almost_equal(cov_matrix, np.diag(expected_variances_list[idx]), ASSERT_NDECIMALS)


def test_financial_features_factory_successful_call():
    feature_list = financial_features_factory(sample_fin_feature_factory_list)

    for feature in feature_list:
        expected_feature = _get_feature_by_name(feature.name, sample_fin_feature_list)
        assert feature.name == expected_feature.name
        assert feature.transformation == expected_feature.transformation
        assert feature.normalization == expected_feature.normalization
        assert feature.nbins == expected_feature.nbins
        assert feature.ndays == expected_feature.ndays
        assert feature.resample_minutes == expected_feature.resample_minutes
        assert feature.start_market_minute == expected_feature.start_market_minute
        assert feature.is_target == expected_feature.is_target


def test_single_financial_features_factory_wrong_keys():
    feature_dict = {
        'name': 'feature1',
        'transformation': {'name': 'log-return'},
        'normalization': None,
        'nbins': 15,
        'ndays': 5,
        'wrong': 1,
        'is_target': False,
    }
    with pytest.raises(KeyError):
        single_financial_feature_factory(feature_dict)

    feature_dict = {
        'name': 'feature1',
        'transformation': {'name': 'wrong'},
        'normalization': 'robust',
        'nbins': 15,
        'ndays': 5,
        'start_market_minute': 1,
        'is_target': False,
    }

    with pytest.raises(AssertionError):
        financial_features_factory(feature_dict)

    feature_dict = {
        'name': 'feature1',
        'transformation': {'name': 'log-return'},
        'normalization': 'wrong',
        'nbins': 15,
        'ndays': 5,
        'start_market_minute': 1,
        'is_target': False,
    }

    with pytest.raises(AssertionError):
        financial_features_factory(feature_dict)


def test_financial_features_factory_wrong_input_type():
    feature_list = {}
    with pytest.raises(AssertionError):
        financial_features_factory(feature_list)


def _get_feature_by_name(name, feature_list):
    for feature in feature_list:
        if feature.name == name:
            return feature
    raise ValueError


def test_get_feature_names():
    assert set(get_feature_names(sample_fin_feature_list)) == {'open', 'close', 'high'}


def test_get_feature_max_ndays():
    assert get_feature_max_ndays(sample_fin_feature_list) == 10
