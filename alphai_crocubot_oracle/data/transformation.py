from abc import ABCMeta, abstractmethod
from datetime import timedelta

import numpy as np
import pandas_market_calendars as mcal
import pandas as pd

from alphai_crocubot_oracle.data import MINUTES_IN_TRADING_DAY
from alphai_crocubot_oracle.data.feature import FinancialFeature, get_feature_names, get_feature_max_ndays

TOTAL_TICKS_FINANCIAL_FEATURES = ['open_value', 'high_value', 'low_value', 'close_value', 'volume_value']
TOTAL_TICKS_M1_FINANCIAL_FEATURES = ['open_log-return', 'high_log-return', 'low_log-return', 'close_log-return',
                                     'volume_log-return']


class DataTransformation(metaclass=ABCMeta):
    @abstractmethod
    def create_train_data(self, *args):
        raise NotImplementedError()

    @abstractmethod
    def create_predict_data(self, *args):
        raise NotImplementedError()


class FinancialDataTransformation(DataTransformation):
    def __init__(self, configuration):
        """
        :param dict configuration: dictionary containing the feature details.
            list feature_config_list: list of dictionaries containing feature details.
            str exchange_name: name of the reference exchange
            int features_ndays: number of trading days worth of data the feature should use.
            int features_resample_minutes: resampling frequency in number of minutes.
            int features_start_market_minute: number of minutes after market open the data collection should start from
            int prediction_frequency_ndays: prediction frequency in number of days
            int prediction_market_minute: number of minutes after market open for the prediction timestamp
            int target_delta_ndays: target time horizon in number of days
            int target_market_minute: number of minutes after market open for the target timestamp
        """
        self._assert_input(configuration)
        self.exchange_calendar = mcal.get_calendar(configuration['exchange_name'])
        self.features_ndays = configuration['features_ndays']
        self.features_resample_minutes = configuration['features_resample_minutes']
        self.features_start_market_minute = configuration['features_start_market_minute']
        self.prediction_frequency_ndays = configuration['prediction_frequency_ndays']
        self.prediction_market_minute = configuration['prediction_market_minute']
        self.target_delta_ndays = configuration['target_delta_ndays']
        self.target_market_minute = configuration['target_market_minute']
        self.features = self._financial_features_factory(configuration['feature_config_list'])

    @staticmethod
    def _assert_input(configuration):
        assert isinstance(configuration['exchange_name'], str)
        assert isinstance(configuration['features_ndays'], int) and configuration['features_ndays'] >= 0
        assert isinstance(configuration['features_resample_minutes'], int) \
            and configuration['features_resample_minutes'] >= 0
        assert isinstance(configuration['features_start_market_minute'], int)
        assert configuration['features_start_market_minute'] < MINUTES_IN_TRADING_DAY
        assert configuration['prediction_frequency_ndays'] >= 0
        assert configuration['prediction_market_minute'] >= 0
        assert configuration['prediction_market_minute'] < MINUTES_IN_TRADING_DAY
        assert configuration['target_delta_ndays'] >= 0
        assert configuration['target_market_minute'] >= 0
        assert configuration['target_market_minute'] < MINUTES_IN_TRADING_DAY
        assert isinstance(configuration['feature_config_list'], list)
        n_targets = 0
        for single_feature_dict in configuration['feature_config_list']:
            if single_feature_dict['is_target']:
                n_targets += 1
        assert n_targets == 1

    def get_total_ticks_x(self):
        """
        Calculate expected total ticks for x data
        :return int: expected total number of ticks for x data
        """
        ticks_in_a_day = np.floor(MINUTES_IN_TRADING_DAY / self.features_resample_minutes) + 1
        intra_day_ticks = np.floor((self.prediction_market_minute - self.features_start_market_minute) /
                                   self.features_resample_minutes)
        total_ticks = ticks_in_a_day * self.features_ndays + intra_day_ticks + 1
        return int(total_ticks)

    def check_x_batch_dimensions(self, feature_x_dict):
        """
        Evaluate if the x batch has the expected dimensions.
        :param dict feature_x_dict: batch of x-features
        :return bool: False if the dimensions are not those expected
        """
        correct_dimensions = True
        for feature_full_name, feature_array in feature_x_dict.items():
            if feature_full_name in TOTAL_TICKS_FINANCIAL_FEATURES:
                if feature_array.shape[0] != self.get_total_ticks_x():
                    correct_dimensions = False
            elif feature_full_name in TOTAL_TICKS_M1_FINANCIAL_FEATURES:
                if feature_array.shape[0] != self.get_total_ticks_x() - 1:
                    correct_dimensions = False
        return correct_dimensions

    def _financial_features_factory(self, feature_config_list):
        """
        Build list of financial features from list of incomplete feature-config dictionaries (class-specific).
        :param list feature_config_list: list of dictionaries containing feature details.
        :return list: list of FinancialFeature objects
        """
        assert isinstance(feature_config_list, list)

        feature_list = []
        for single_feature_dict in feature_config_list:
            feature_list.append(FinancialFeature(
                single_feature_dict['name'],
                single_feature_dict['transformation'],
                single_feature_dict['normalization'],
                single_feature_dict['nbins'],
                self.features_ndays,
                self.features_resample_minutes,
                self.features_start_market_minute,
                single_feature_dict['is_target'],
                self.exchange_calendar
            ))

        return feature_list

    def _get_market_open_list(self, raw_data_dict):
        """
        Return a list of market open timestamps from input data_dict
        :param dict raw_data_dict: dictionary of dataframes containing features data.
        :return pd.Series: list of market open timestamps.
        """
        features_keys = get_feature_names(self.features)
        raw_data_start_date = raw_data_dict[features_keys[0]].index[0].date()
        raw_data_end_date = raw_data_dict[features_keys[0]].index[-1].date()
        return self.exchange_calendar.schedule(str(raw_data_start_date), str(raw_data_end_date)).market_open

    def get_target_feature(self):
        """
        Return the target feature in self.features
        :return FinancialFeature: target feature
        """
        return [feature for feature in self.features if feature.is_target][0]

    def get_prediction_data_all_features(self, raw_data_dict, prediction_timestamp, universe=None,
                                         target_timestamp=None):
        """
        Collect processed prediction x and y data for all the features.
        :param dict raw_data_dict: dictionary of dataframes containing features data.
        :param Timestamp prediction_timestamp: Timestamp when the prediction is made
        :param list/None universe: list of relevant symbols
        :param Timestamp/None target_timestamp: Timestamp the prediction is for.
        :return (dict, dict): feature_x_dict, feature_y_dict
        """
        feature_x_dict, feature_y_dict = {}, {}

        for feature in self.features:
            if universe is None:
                universe = raw_data_dict[feature.name].columns

            feature_x, feature_y = feature.get_prediction_data(
                raw_data_dict[feature.name][universe],
                prediction_timestamp,
                target_timestamp,
            )
            feature_x_dict[feature.full_name] = feature_x.values

            if feature_y is not None:
                feature_y_dict[feature.full_name] = feature_y.values

        if len(feature_y_dict) > 0:
            assert len(feature_y_dict) == 1, 'Only one target is allowed'
        else:
            feature_y_dict = None

        return feature_x_dict, feature_y_dict

    def create_train_data(self, raw_data_dict, historical_universes):
        """
        Prepare x and y data for training
        :param dict raw_data_dict: dictionary of dataframes containing features data.
        :param pd.Dataframe historical_universes: Dataframe with three columns ['start_date', 'end_date', 'assets']
        :return (dict, dict): feature_x_dict, feature_y_dict
        """

        training_dates = self.get_training_market_dates(raw_data_dict)
        return self.create_data(raw_data_dict, training_dates, historical_universes)

    def create_predict_data(self, raw_data_dict):
        """
        Prepare x data for inference purposes.
        :param dict raw_data_dict: dictionary of dataframes containing features data.
        :return dict: feature_x_dict
        """

        current_market_open = self.get_current_market_date(raw_data_dict)
        predict_x, _ = self.create_data(raw_data_dict, simulated_market_dates=current_market_open)
        return predict_x

    def create_data(self, raw_data_dict, simulated_market_dates, historical_universes=None):
        """
        Create x and y data
        :param dict raw_data_dict: dictionary of dataframes containing features data.
        :param market_open_list:
        :param pd.Dataframe historical_universes: Dataframe with three columns ['start_date', 'end_date', 'assets']
        :return (dict, dict): feature_x_dict, feature_y_dict
        """

        market_open_list = self._get_market_open_list(raw_data_dict)
        data_x_list, data_y_list = [], []

        for prediction_market_open in simulated_market_dates:
            date_index = pd.Index(market_open_list).get_loc(prediction_market_open)
            target_index = date_index + self.target_delta_ndays

            if target_index < len(market_open_list):
                target_market_open = market_open_list[target_index]
            else:
                target_market_open = None

            feature_x_dict, feature_y_dict = self.build_features(raw_data_dict, historical_universes,
                                                                 prediction_market_open, target_market_open)

            if self.check_x_batch_dimensions(feature_x_dict):
                data_x_list.append(feature_x_dict)
                data_y_list.append(feature_y_dict)

        x_dict = self.stack_samples_for_each_feature(data_x_list)

        if target_market_open is None:
            y_dict = None
        else:
            y_dict = self._make_classified_y_dict(data_y_list)

        return x_dict, y_dict

    def _make_classified_y_dict(self, y_list):

        y_dict = self.stack_samples_for_each_feature(y_list)
        target_feature = self.get_target_feature()
        if target_feature.nbins:
            y_key_list = list(y_dict.keys())
            y_train_data = y_dict[y_key_list[0]]
            y_dict = {target_feature.full_name: target_feature.classify_train_data_y(y_train_data)}
        else:
            raise NotImplementedError('If not using a classifier, we need to implement an inverse y transformation.'
                                      ' Take care with the discintion between the '
                                      'timescale for the x and y log returns')
        return y_dict

    def build_features(self, raw_data_dict, historical_universes, prediction_market_open, target_market_open):
        """ Creates dictionaries of features and labels for a single window

        :param dict raw_data_dict: dictionary of dataframes containing features data.
        :param pd.Dataframe historical_universes: Dataframe with three columns ['start_date', 'end_date', 'assets']
        :param prediction_market_open:
        :param target_market_open:
        :return:
        """

        if historical_universes is None:
            universe = None
        else:
            prediction_date = prediction_market_open.date()
            universe = _get_universe_from_date(prediction_date, historical_universes)

        prediction_timestamp = prediction_market_open + timedelta(minutes=self.prediction_market_minute)

        if target_market_open is None:
            target_timestamp = None
        else:
            target_timestamp = target_market_open + timedelta(minutes=self.target_market_minute)

        return self.get_prediction_data_all_features(
            raw_data_dict,
            prediction_timestamp,
            universe,
            target_timestamp,
        )

    @staticmethod
    def stack_samples_for_each_feature(samples):
        if len(samples) == 0:
            raise ValueError("At least one sample required for stacking samples.")

        feature_names = samples[0].keys()

        stacked_samples = {}

        for feature_name in feature_names:
            if len(samples) == 1:
                sample = samples[0]
                stacked_samples[feature_name] = np.expand_dims(sample[feature_name], axis=0)
            else:
                stacked_samples[feature_name] = np.stack([sample[feature_name] for sample in samples])

        return stacked_samples

    def inverse_transform_multi_predict_y(self, predict_y):
        """
        Inverse-transform multi-pass predict_y data
        :param ndarray predict_y: target multi-pass prediction data
        :return ndarray: inversely transformed multi-pass predict_y data
        """
        target_feature = self.get_target_feature()
        means, cov_matrix = target_feature.inverse_transform_multi_predict_y(predict_y)

        return means, cov_matrix

    def get_current_market_date(self, raw_data_dict):
        return [self._get_market_open_list(raw_data_dict)[-1]]

    def get_training_market_dates(self, raw_data_dict):
        """ Returns all dates on which we have both x and y data"""

        max_feature_ndays = get_feature_max_ndays(self.features)

        return self._get_market_open_list(raw_data_dict)[max_feature_ndays:-self.target_delta_ndays]


def _get_universe_from_date(date, historical_universes):
    """
    Select the universe list of symbols from historical_universes dataframe, given input date.
    :param pd.datetime.date date: Date for which the universe is required.
    :param pd.Dataframe historical_universes: Dataframe with three columns ['start_date', 'end_date', 'assets']
    :return list: list of relevant symbols
    """
    universe_idx = historical_universes[(date >= historical_universes.start_date) &
                                        (date < historical_universes.end_date)].index[0]
    return historical_universes.assets[universe_idx]
