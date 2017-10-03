from copy import deepcopy
from datetime import timedelta

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from sklearn.preprocessing import QuantileTransformer

from alphai_crocubot_oracle.data import FINANCIAL_FEATURE_TRANSFORMATIONS, FINANCIAL_FEATURE_NORMALIZATIONS, \
    MINUTES_IN_TRADING_DAY, MARKET_DAYS_SEARCH_MULTIPLIER, MIN_MARKET_DAYS_SEARCH

from alphai_crocubot_oracle.data.classifier import BinDistribution, classify_labels, declassify_labels


class FinancialFeature(object):
    def __init__(self, name, transformation, normalization, nbins, ndays, resample_minutes, start_market_minute,
                 is_target, exchange_calendar):
        """
        Object containing all the information to manipulate the data relative to a financial feature.
        :param str name: Name of the feature
        :param dict transformation: contains name and parameters to use for processing, name must be in
            FINANCIAL_FEATURE_TRANSFORMATIONS
        :param str/None normalization: type of normalization. Can be None.
        :param int/None nbins: number of bins to be used for target classification. Can be None.
        :param int ndays: number of trading days worth of data the feature should use.
        :param int resample_minutes: resampling frequency in number of minutes.
        :param int start_market_minute: number of minutes after market open the data collection should start from.
        :param bool is_target: if True the feature is a target.
        :param pandas_market_calendar exchange_calendar: exchange calendar.
        """
        self._assert_input(name, transformation, normalization, nbins, ndays, resample_minutes, start_market_minute,
                           is_target)
        self.name = name
        self.transformation = transformation
        self.normalization = normalization
        self.nbins = nbins
        self.ndays = ndays
        self.resample_minutes = resample_minutes
        self.start_market_minute = start_market_minute
        self.is_target = is_target
        self.exchange_calendar = exchange_calendar
        self.n_series = None

        self.bin_distribution = None
        self.has_fitted_scaler = False
        self.classify_per_series = False

        if self.normalization:
            if self.normalization == 'robust':
                self.scaler = RobustScaler()
            elif self.normalization == 'min_max':
                self.scaler = MinMaxScaler()
            elif self.normalization == 'standard':
                self.scaler = StandardScaler()
            elif self.normalization == 'gaussian':
                self.scaler = QuantileTransformer(output_distribution='normal')
            else:
                raise NotImplementedError('Requested normalisation not supported: {}'.format(self.normalization))
        else:
            self.scaler = None

    @property
    def full_name(self):
        return '{}_{}'.format(self.name, self.transformation['name'])

    @staticmethod
    def _assert_input(name, transformation, normalization, nbins, ndays, resample_minutes, start_market_minute,
                      is_target):
        assert isinstance(name, str)
        assert isinstance(transformation, dict)
        assert 'name' in transformation, 'The transformation dict does not contain the key "name"'
        assert transformation['name'] in FINANCIAL_FEATURE_TRANSFORMATIONS
        assert normalization in FINANCIAL_FEATURE_NORMALIZATIONS
        assert (isinstance(nbins, int) and nbins > 0) or nbins is None
        assert isinstance(ndays, int) and ndays >= 0
        assert isinstance(resample_minutes, int) and resample_minutes >= 0
        assert isinstance(start_market_minute, int)
        assert start_market_minute < MINUTES_IN_TRADING_DAY
        assert isinstance(is_target, bool)
        if transformation['name'] == 'ewma':
            assert 'halflife' in transformation
        if transformation['name'] == 'KER':
            assert 'lag' in transformation

    def process_prediction_data_x(self, prediction_data_x):
        """
        Apply feature-specific transformations to input prediction_data_x
        :param pd.Dataframe prediction_data_x: X data for model prediction task
        :return pd.Dataframe: processed_prediction_data_x
        """
        assert isinstance(prediction_data_x, pd.DataFrame)
        processed_prediction_data_x = deepcopy(prediction_data_x)

        if self.transformation['name'] == 'log-return':
            processed_prediction_data_x = np.log(processed_prediction_data_x.pct_change() + 1). \
                replace([np.inf, -np.inf], np.nan).dropna()

        if self.transformation['name'] == 'stochastic_k':

            columns = processed_prediction_data_x.columns
            processed_prediction_data_x \
                = ((processed_prediction_data_x.iloc[-1] - processed_prediction_data_x.min()) /
                   (processed_prediction_data_x.max() - processed_prediction_data_x.min())) * 100.

            processed_prediction_data_x = np.expand_dims(processed_prediction_data_x, axis=0)
            processed_prediction_data_x = pd.DataFrame(processed_prediction_data_x, columns=columns)

        if self.transformation['name'] == 'ewma':
            processed_prediction_data_x = \
                processed_prediction_data_x.ewm(halflife=self.transformation['halflife']).mean()

        if self.transformation['name'] == 'KER':
            direction = processed_prediction_data_x.diff(self.transformation['lag']).abs()
            volatility = processed_prediction_data_x.diff().abs().rolling(window=self.transformation['lag']).sum()

            direction.dropna(axis=0, inplace=True)
            volatility.dropna(axis=0, inplace=True)

            assert direction.shape == volatility.shape, ' direction and volatility need same shape in KER'

            processed_prediction_data_x = direction / volatility
            processed_prediction_data_x.dropna(axis=0, inplace=True)

        if self.has_fitted_scaler:
            processed_prediction_data_x.loc[:, :] = self.scaler.transform(processed_prediction_data_x)
        elif self.scaler is not None:
            processed_prediction_data_x.loc[:, :] = self.scaler.fit_transform(processed_prediction_data_x)
            self.has_fitted_scaler = True

        return processed_prediction_data_x

    def fit_normalisation_constants(self, data_x):
        """ Compute normalisation across the entire training set"""

        if self.scaler is not None:
            data_x.loc[:, :] = self.scaler.fit_transform(data_x)
            self.has_fitted_scaler = True

    def process_prediction_data_y(self, prediction_data_y, prediction_reference_data):
        """
        Apply feature-specific transformations to input prediction_data_y
        :param pd.Series prediction_data_y: y data for model prediction task
        :param pd.Series prediction_reference_data: reference data-point to calculate differential metrics
        :return pd.Series: processed_prediction_data_y
        """
        assert self.is_target
        assert isinstance(prediction_data_y, pd.Series)
        processed_prediction_data_y = deepcopy(prediction_data_y)

        if self.transformation['name'] == 'log-return':
            processed_prediction_data_y = np.log(prediction_data_y / prediction_reference_data). \
                replace([np.inf, -np.inf], np.nan).dropna()

        if self.scaler:
            if self.nbins is None:
                raise NotImplementedError('y scaling is not required for classifiers, but is required for regression')

        return processed_prediction_data_y

    def _get_safe_schedule_start_date(self, prediction_timestamp):
        """
        Calculate a safe schedule start date from input timestamp so that at least self.ndays trading days are available
        :param Timestamp prediction_timestamp: Timestamp when the prediction is made
        :return Timestamp: schedule_start_date
        """
        safe_ndays = max(MIN_MARKET_DAYS_SEARCH, MARKET_DAYS_SEARCH_MULTIPLIER * self.ndays)
        return prediction_timestamp - timedelta(days=safe_ndays)

    def _get_start_timestamp_x(self, prediction_timestamp):
        """
        Calculate the start timestamp of x-data for a given prediction timestamp.
        :param Timestamp prediction_timestamp: Timestamp when the prediction is made
        :return Timestamp: start timestamp of x-data
        """
        schedule_start_date = str(self._get_safe_schedule_start_date(prediction_timestamp))
        schedule_end_date = str(prediction_timestamp.date())
        market_open_list = self.exchange_calendar.schedule(schedule_start_date, schedule_end_date).market_open
        prediction_market_open = market_open_list[prediction_timestamp.date()]
        prediction_market_open_idx = np.argwhere(market_open_list == prediction_market_open).flatten()[0]
        start_timestamp_x = market_open_list[prediction_market_open_idx - self.ndays] + timedelta(
            minutes=self.start_market_minute)
        return start_timestamp_x

    def _index_selection_x(self, date_time_index, prediction_timestamp):
        """
        Create index selection rule for x data
        :param Timestamp prediction_timestamp: Timestamp when the prediction is made
        :return: index selection rule
        """
        start_timestamp_x = self._get_start_timestamp_x(prediction_timestamp)
        return (date_time_index >= start_timestamp_x) & (date_time_index <= prediction_timestamp)

    def _select_prediction_data_x(self, data_frame, prediction_timestamp):
        """
        Select the x-data relevant for a input prediction timestamp.
        :param pd.Dataframe data_frame: raw x-data (unselected, unprocessed)
        :param Timestamp prediction_timestamp: Timestamp when the prediction is made
        :return pd.Dataframe: selected x-data (unprocessed)
        """
        prediction_index_selection_x = self._index_selection_x(data_frame.index, prediction_timestamp)
        return data_frame[prediction_index_selection_x]

    def get_prediction_data(self, data_frame, prediction_timestamp, target_timestamp=None):
        """
        Calculate x and y data for prediction. y-data will be None if target_timestamp is None.
        :param pd.Dataframe data_frame: raw data (unselected, unprocessed).
        :param Timestamp prediction_timestamp: Timestamp when the prediction is made
        :param Timestamp target_timestamp: Timestamp the prediction is for.
        :return (pd.Dataframe, pd.Dataframe): prediction_data_x, prediction_data_y (selected and processed)
        """
        selected_prediction_data_x = self._select_prediction_data_x(data_frame, prediction_timestamp)
        prediction_data_x = self.process_prediction_data_x(selected_prediction_data_x)

        prediction_data_y = None
        if self.is_target and target_timestamp is not None:
            prediction_data_y = self.process_prediction_data_y(
                data_frame.loc[target_timestamp],
                data_frame.loc[prediction_timestamp],
            )

        return prediction_data_x, prediction_data_y

    def calculate_bin_distribution(self, train_y):
        """
        Calculate bin distribution from training target values.
        :param ndarray train_y: Training target labels to calculate bin distribution.
        :return: Nothing.
        """
        assert isinstance(self.nbins, int) and self.nbins > 0

        if self.classify_per_series:
            self.bin_distribution = []
            for i in range(self.n_series):
                self.bin_distribution.append(BinDistribution(train_y[i, :], self.nbins))
        else:
            self.bin_distribution = BinDistribution(train_y, self.nbins)

    def classify_train_data_y(self, train_y):
        """
        Classify training target values.
        :param ndarray train_y: Training target labels to calculate bin distribution. Of shape (batch_size, n_series)
        :return ndarray: classified train_y. Of shape (batch_size, n_series, n_bins)
        """

        if self.nbins is None:
            return train_y

        self.bin_distribution = None
        batch_size = train_y.shape[0]
        self.n_series = train_y.shape[1]
        self.calculate_bin_distribution(train_y)

        if self.classify_per_series:
            labels = np.zeros((batch_size, self.n_series, self.nbins))
            for i in range(self.n_series):
                bin_edges = self.bin_distribution[i].bin_edges
                labels[:, i, :] = classify_labels(bin_edges, train_y[:, i])
                # FIXME May have to use swapaxes if this assignment to dims 0,2 doesnt work
        else:
            labels = classify_labels(self.bin_distribution.bin_edges, train_y)

        return labels

    def declassify_single_predict_y(self, predict_y):
        raise NotImplementedError('Declassification is only available for multi-pass prediction at the moment.')

    def inverse_transform_single_predict_y(self, predict_y):
        """
        Inverse-transform single-pass predict_y data
        :param pd.Dataframe predict_y: target single-pass prediction
        :return pd.Dataframe: inversely transformed single-pass predicted_y data
        """
        assert self.is_target
        if self.normalization:
            inverse_transf_predicted_y = self.scaler.inverse_transform(predict_y)
        else:
            inverse_transf_predicted_y = predict_y

        return inverse_transf_predicted_y

    def declassify_multi_predict_y(self, predict_y):
        """
        Declassify multi-pass predict_y data
        :param predict_y: target multi-pass prediction with axes (passes, series, bins)
        :return: mean and variance of target multi-pass prediction
        """
        n_series = predict_y.shape[1]

        if self.nbins:
            means = np.zeros(shape=(n_series,))
            variances = np.zeros(shape=(n_series,))
            for series_idx in range(n_series):
                if self.classify_per_series:
                    series_bins = self.bin_distribution[series_idx]
                else:
                    series_bins = self.bin_distribution

                means[series_idx], variances[series_idx] = \
                    declassify_labels(series_bins, predict_y[:, series_idx, :])
        else:
            means = np.mean(predict_y, axis=0)
            variances = np.var(predict_y, axis=0)

        return means, variances

    def inverse_transform_multi_predict_y(self, predict_y):
        """
        Inverse-transform multi-pass predict_y data
        :param pd.Dataframe predict_y: target multi-pass prediction
        :return pd.Dataframe: inversely transformed mean and variance of target multi-pass prediction
        """
        assert self.is_target
        means, variances = self.declassify_multi_predict_y(predict_y)

        if self.normalization:
            if self.normalization == 'standard':
                inverse_transf_means = means + self.scaler.mean_
                inverse_transf_variances = variances * self.scaler.var_
            else:
                raise NotImplementedError('Requested normalisation cannot be inverted')
        else:
            inverse_transf_means = means
            inverse_transf_variances = variances

        diag_cov_matrix = np.diag(inverse_transf_variances)

        return inverse_transf_means, diag_cov_matrix


def single_financial_feature_factory(feature_config):
    """
    Build target financial feature from dictionary.
    :param dict feature_config: dictionary containing feature details.
    :return FinancialFeature: FinancialFeature object
    """
    assert isinstance(feature_config, dict)

    return FinancialFeature(
        feature_config['name'],
        feature_config['transformation'],
        feature_config['normalization'],
        feature_config['nbins'],
        feature_config['ndays'],
        feature_config['resample_minutes'],
        feature_config['start_market_minute'],
        feature_config['is_target'],
        mcal.get_calendar(feature_config['exchange_name']))


def financial_features_factory(feature_config_list):
    """
    Build list of financial features from list of complete feature-config dictionaries.
    :param list feature_config_list: list of dictionaries containing feature details.
    :return list: list of FinancialFeature objects
    """
    assert isinstance(feature_config_list, list)

    feature_list = []
    for single_feature_dict in feature_config_list:
        feature_list.append(single_financial_feature_factory(single_feature_dict))

    return feature_list


def get_feature_names(feature_list):
    """
    Return unique names of feature list
    :param list feature_list: list of Feature objects
    :return list: list of strings
    """
    return list(set([feature.name for feature in feature_list]))


def get_feature_max_ndays(feature_list):
    """
    Return max ndays of feature list
    :param list feature_list: list of Feature objects
    :return int: max ndays of feature list
    """
    return max([feature.ndays for feature in feature_list])
