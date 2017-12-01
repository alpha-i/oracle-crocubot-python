from copy import deepcopy
import logging

import numpy as np
import pandas as pd
import pandas_market_calendars as mcal
from sklearn.preprocessing import RobustScaler, MinMaxScaler, StandardScaler
from sklearn.preprocessing import QuantileTransformer

from alphai_crocubot_oracle.data import FINANCIAL_FEATURE_TRANSFORMATIONS, FINANCIAL_FEATURE_NORMALIZATIONS

from alphai_crocubot_oracle.data.classifier import BinDistribution, classify_labels, declassify_labels

SCIKIT_SHAPE = (-1, 1)
logging.getLogger(__name__).addHandler(logging.NullHandler())


class FinancialFeature(object):
    def __init__(self, name, transformation, normalization, nbins, length,
                 is_target, exchange_calendar, classify_per_series=False, normalise_per_series=False):
        """
        Object containing all the information to manipulate the data relative to a financial feature.
        :param str name: Name of the feature
        :param dict transformation: contains name and parameters to use for processing, name must be in
            FINANCIAL_FEATURE_TRANSFORMATIONS
        :param str/None normalization: type of normalization. Can be None.
        :param int/None nbins: number of bins to be used for target classification. Can be None.
        :param int length: expected number of elements in the feature
        :param bool is_target: if True the feature is a target.
        :param pandas_market_calendar exchange_calendar: exchange calendar.
        """
        # FIXME the get_default_flags args are temporary. We need to load a get_default_flags config in the unit tests.

        self._assert_input(name, transformation, normalization, nbins, length, is_target)
        self.name = name
        self.transformation = transformation
        self.normalization = normalization
        self.nbins = nbins
        self.is_target = is_target
        self.exchange_calendar = exchange_calendar
        self.n_series = None
        self.length = length

        self.bin_distribution = None
        if self.nbins:
            self.bin_distribution_dict = {}
        else:
            self.bin_distribution_dict = None

        self.classify_per_series = classify_per_series
        self.normalise_per_series = normalise_per_series

        if self.normalization:
            self.scaler_dict = {}
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
            self.scaler_dict = None

    @property
    def full_name(self):
        return '{}_{}'.format(self.name, self.transformation['name'])

    @staticmethod
    def _assert_input(name, transformation, normalization, nbins, length, is_target):
        assert isinstance(name, str)
        assert isinstance(transformation, dict)
        assert 'name' in transformation, 'The transformation dict does not contain the key "name"'
        assert transformation['name'] in FINANCIAL_FEATURE_TRANSFORMATIONS
        assert normalization in FINANCIAL_FEATURE_NORMALIZATIONS
        assert (isinstance(nbins, int) and nbins > 0) or nbins is None
        assert (isinstance(length, int) and length > 0)
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
                replace([np.inf, -np.inf], np.nan)

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

        return processed_prediction_data_x.iloc[1:]  # Discard first element, often an unwanted zero or nan

    def fit_normalisation(self, symbol_data, symbol=None):
        """ Creates a scikitlearn scalar, assigns it to a dictionary, fits it to the data

        :param symbol:
        :param symbol_data:
        :return:
        """

        symbol_data.flatten()
        symbol_data = symbol_data[np.isfinite(symbol_data)]
        symbol_data = reshape_scikit(symbol_data)  # Reshape for scikitlearn

        if symbol:
            self.scaler_dict[symbol] = deepcopy(self.scaler)
            self.scaler_dict[symbol].fit(symbol_data)
        else:
            self.scaler.fit(symbol_data)

    def apply_normalisation(self, dataframe):
        """ Compute normalisation across the entire training set, or apply predetermined normalistion to prediction.

        :param pd dataframe: Features of shape [n_samples, n_series, n_features]
        :return:
        """

        for symbol in dataframe:
            data_x = dataframe[symbol].values
            original_shape = data_x.shape
            data_x = data_x.flatten()
            valid_data = data_x[np.isfinite(data_x)]
            flat_shape = valid_data.shape

            if len(valid_data) > 0:
                scaler = self.get_scaler(symbol)

                if scaler is None:
                    logging.warning("Symbol lacks normalisation scaler: {}. Dropping from dataframe.".format(symbol))
                    dataframe.drop(symbol, axis=1, inplace=True)
                else:
                    valid_data = valid_data.reshape(SCIKIT_SHAPE)
                    try:
                        scaler.transform(valid_data, copy=False)
                    except:  # Some scalers cannot be performed in situ
                        valid_data = scaler.transform(valid_data)
                    data_x[np.isfinite(data_x)] = valid_data.reshape(flat_shape)
                    dataframe[symbol] = data_x.reshape(original_shape)

        return dataframe

    def get_scaler(self, symbol):
        """ Returns scaler for a given symbol

        :param str symbol: The time series we wish to transform
        :return: scikit scaler: The scikitlearn scaler
        """

        if self.normalise_per_series:
            scaler = self.scaler_dict.get(symbol, None)
        else:
            scaler = self.scaler

        return scaler

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
                replace([np.inf, -np.inf], np.nan)

        if self.scaler:
            if self.nbins is None:
                raise NotImplementedError('y scaling is not required for classifiers, but is required for regression')

        return processed_prediction_data_y

    def _select_prediction_data_x(self, data_frame, prediction_timestamp):
        """
        Select the x-data relevant for a input prediction timestamp.
        :param pd.Dataframe data_frame: raw x-data (unselected, unprocessed)
        :param Timestamp prediction_timestamp: Timestamp when the prediction is made
        :return pd.Dataframe: selected x-data (unprocessed)
        """

        try:
            end_point = data_frame.index.get_loc(prediction_timestamp, method='pad')
            end_index = end_point + 1  # +1 because iloc is not inclusive of end index
            start_index = end_point - self.length
        except:
            logging.warning('Prediction timestamp {} not within range of dataframe'.format(prediction_timestamp))
            start_index = 0
            end_index = -1

        return data_frame.iloc[start_index:end_index, :]

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

    def fit_classification(self, symbol, symbol_data):
        """  Fill dict with classifiers

        :param symbol_data:
        :return:
        """

        if self.nbins is None:
            return

        self.bin_distribution_dict[symbol] = BinDistribution(symbol_data, self.nbins)

    def apply_classification(self, dataframe):
        """ Apply predetermined classification to y data.

        :param pd dataframe data_x: Features of shape [n_samples, n_series, n_features]
        :return:
        """

        hot_dataframe = pd.DataFrame(0, index=np.arange(self.nbins), columns=dataframe.columns)

        for symbol in dataframe:
            data_y = dataframe[symbol].values

            if symbol in self.bin_distribution_dict:
                symbol_binning = self.bin_distribution_dict[symbol]
                one_hot_labels = classify_labels(symbol_binning.bin_edges, data_y)
                if one_hot_labels.shape[-1] > 1:
                    hot_dataframe[symbol] = np.squeeze(one_hot_labels)
            else:
                logging.warning("Symbol lacks clasification bins: {}".format(symbol))
                dataframe.drop(symbol, axis=1, inplace=True)
                logging.warning("Dropping {} from dataframe.".format(symbol))

        return hot_dataframe

    def inverse_transform_multi_predict_y(self, predict_y, symbols):
        """
        Inverse-transform multi-pass predict_y data
        :param pd.Dataframe predict_y: target multi-pass prediction
        :return pd.Dataframe: inversely transformed mean and variance of target multi-pass prediction
        """
        assert self.is_target

        n_symbols = len(symbols)
        print("new symbols:", n_symbols)
        means = np.zeros(shape=(n_symbols,))
        variances = np.zeros(shape=(n_symbols,))
        assert predict_y.shape[1] == n_symbols, "Weird shape - predict y not equal to n symbols"

        for i, symbol in enumerate(symbols):
            if symbol in self.bin_distribution_dict:
                symbol_bins = self.bin_distribution_dict[symbol]
                means[i], variances[i] = declassify_labels(symbol_bins, predict_y[:, i, :])
            else:
                logging.warning("No bin distribution found for symbol: {}".format(symbol))
                means[i] = np.nan
                variances[i] = np.nan

        variances[variances == 0] = 1.0  # FIXME Hack

        diag_cov_matrix = np.diag(variances)
        return means, diag_cov_matrix


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
        feature_config['length'],
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


def reshape_scikit(data_x):
    """ Scikit expects an input of the form [samples, features]; normalisation applied separately to each feature.

    :param data_x: Features of any shape
    :return: nparray Same data as input, but now with two dimensions: [samples, f], each f has own normalisation
    """

    return data_x.reshape(SCIKIT_SHAPE)