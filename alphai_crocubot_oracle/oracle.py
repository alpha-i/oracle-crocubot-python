# Interface with quant workflow.
# Trains the network then uses it to make predictions
# Also transforms the data before and after the predictions are made

# A fairly generic interface, in that it can easily applied to other models

import logging
from timeit import default_timer as timer
from copy import deepcopy

import numpy as np
import pandas as pd

from alphai_crocubot_oracle.crocubot.helpers import TensorflowPath, TensorboardOptions
from alphai_crocubot_oracle.data.providers import TrainDataProvider
from alphai_feature_generation.transformation import FinancialDataTransformation
from alphai_time_series.transform import gaussianise

import alphai_crocubot_oracle.crocubot.train as crocubot
import alphai_crocubot_oracle.crocubot.evaluate as crocubot_eval
import alphai_crocubot_oracle.dropout.train as dropout
import alphai_crocubot_oracle.dropout.evaluate as dropout_eval
from alphai_crocubot_oracle.flags import build_tensorflow_flags
import alphai_crocubot_oracle.topology as tp
from alphai_crocubot_oracle import DATETIME_FORMAT_COMPACT
from alphai_crocubot_oracle.covariance import estimate_covariance
from alphai_crocubot_oracle.helpers import TrainFileManager, logtime

CLIP_VALUE = 5.0  # Largest number allowed to enter the network
DEFAULT_N_CORRELATED_SERIES = 5
DEFAULT_N_CONV_FILTERS = 5
DEFAULT_CONV_KERNEL_SIZE = [3, 3, 1]
FEATURE_TO_RANK_CORRELATIONS = 0  # Use the first feature to form correlation coefficients
TRAIN_FILE_NAME_TEMPLATE = "{}_train_crocubot"
DEFAULT_NETWORK = 'crocubot'

logging.getLogger(__name__).addHandler(logging.NullHandler())


class CrocubotOracle:
    def __init__(self, configuration):
        """
        :param configuration: Dictionary containing all the parameters. Full specifications can be found at:
        oracle-crocubot-python/docs/crocubot_options.md
        """

        self.network = configuration.get('network', DEFAULT_NETWORK)
        logging.info('Initialising {} oracle.'.format(self.network))

        configuration = self.update_configuration(configuration)
        feature_list = configuration['data_transformation']['feature_config_list']

        self._data_transformation = FinancialDataTransformation(configuration['data_transformation'])
        self._train_path = configuration['train_path']
        self._covariance_method = configuration['covariance_method']
        self._covariance_ndays = configuration['covariance_ndays']
        self._n_features = len(feature_list)

        self.use_historical_covariance = configuration.get('use_historical_covariance', False)
        n_correlated_series = configuration.get('n_correlated_series', DEFAULT_N_CORRELATED_SERIES)

        self._configuration = configuration
        self._train_file_manager = TrainFileManager(
            self._train_path,
            self._get_train_template(),
            DATETIME_FORMAT_COMPACT
        )

        self._train_file_manager.ensure_path_exists()
        self._est_cov = None

        self._tensorflow_flags = build_tensorflow_flags(configuration)  # Perhaps use separate config dict here?

        if self._tensorflow_flags.predict_single_shares:
            self._n_input_series = int(np.minimum(n_correlated_series, configuration['n_series']))
            self._n_forecasts = 1
        else:
            self._n_input_series = configuration['n_series']
            self._n_forecasts = configuration['n_forecasts']

        self._topology = None

    def train(self, historical_universes, train_data, execution_time):
        """
        Trains the model

        :param pd.DataFrame historical_universes: dates and symbols of historical universes
        :param dict train_data: OHLCV data as dictionary of pandas DataFrame.
        :param datetime.datetime execution_time: time of execution of training
        :return:
        """
        logging.info('Training model on {}.'.format(
            execution_time,
        ))

        self.verify_pricing_data(train_data)
        train_x_dict, train_y_dict = self._data_transformation.create_train_data(train_data, historical_universes)

        logging.info("Preprocessing training data")
        train_x = self._preprocess_inputs(train_x_dict)
        train_y = self._preprocess_outputs(train_y_dict)
        logging.info("Processed train_x shape {}".format(train_x.shape))
        train_x, train_y = self.filter_nan_samples(train_x, train_y)
        logging.info("Filtered train_x shape {}".format(train_x.shape))
        n_valid_samples = train_x.shape[0]

        if n_valid_samples == 0:
            raise ValueError("Aborting training: No valid samples")
        elif n_valid_samples < 2e4:
            logging.warning("Low number of training samples: {}".format(n_valid_samples))

        # Topology can either be directly constructed from layers, or build from sequence of parameters
        if self._topology is None:
            n_timesteps = train_x.shape[2]
            self.initialise_topology(n_timesteps)

        logging.info('Initialised network topology: {}.'.format(self._topology.layers))

        logging.info('Training features of shape: {}.'.format(
            train_x.shape,
        ))
        logging.info('Training labels of shape: {}.'.format(
            train_y.shape,
        ))

        resume_train_path = None

        if self._tensorflow_flags.resume_training:
            try:
                resume_train_path = self._train_file_manager.latest_train_filename(execution_time)
            except ValueError:
                pass

        train_path = self._train_file_manager.new_filename(execution_time)

        tensorflow_path = TensorflowPath(train_path, resume_train_path)
        tensorboard_options = TensorboardOptions(self._tensorflow_flags.tensorboard_log_path,
                                                 self._tensorflow_flags.learning_rate,
                                                 self._tensorflow_flags.batch_size,
                                                 execution_time
                                                 )

        first_sample = train_x[0, :].flatten()
        logging.info("Sample from first example in train_x: {}".format(first_sample[0:8]))
        data_provider = TrainDataProvider(train_x, train_y, self._tensorflow_flags.batch_size)
        self._do_train(tensorflow_path, tensorboard_options, data_provider)

    @logtime(message="Training the model.")
    def _do_train(self, tensorflow_path, tensorboard_options, data_provider):
        if self.network == 'crocubot':
            crocubot.train(self._topology, data_provider, tensorflow_path, tensorboard_options, self._tensorflow_flags)
        elif self.network == 'dropout':
            dropout.train(data_provider, tensorflow_path, self._tensorflow_flags)
        elif self.network == 'inception':
            raise NotImplementedError('Requested network not supported:', self.network)
            # inception.train(data_provider, tensorflow_path, self._tensorflow_flags)
        else:
            raise NotImplementedError('Requested network not supported:', self.network)

    def _get_train_template(self):
        return "{}_train_" + self.network

    def predict(self, predict_data, execution_time):
        """

        :param dict predict_data: OHLCV data as dictionary of pandas DataFrame
        :param datetime.datetime execution_time: time of execution of prediction

        :return : mean vector (pd.Series) and two covariance matrices (pd.DF)
        """

        if self._topology is None:
            logging.warning('Not ready for prediction - safer to run train first')

        logging.info('Oracle prediction on {}.'.format(execution_time))

        self.verify_pricing_data(predict_data)
        latest_train_file = self._train_file_manager.latest_train_filename(execution_time)
        predict_x, symbols, predict_timestamp, target_timestamp = \
            self._data_transformation.create_predict_data(predict_data)

        logging.info('Predicting mean values.')
        start_time = timer()
        predict_x = self._preprocess_inputs(predict_x)

        if self._topology is None:
            n_timesteps = predict_x.shape[2]
            self.initialise_topology(n_timesteps)

        if self.network == 'crocubot':
            # Verify data is the correct shape
            network_input_shape = self._topology.get_network_input_shape()
            data_input_shape = predict_x.shape[-3:]

            if data_input_shape != network_input_shape:
                err_msg = 'Data shape' + str(data_input_shape) + " doesnt match network input " + str(
                    network_input_shape)
                raise ValueError(err_msg)

            predict_y = crocubot_eval.eval_neural_net(
                predict_x, self._topology,
                self._tensorflow_flags,
                latest_train_file
            )
        else:
            predict_y = dropout_eval.eval_neural_net(predict_x, self._tensorflow_flags, latest_train_file)

        end_time = timer()
        eval_time = end_time - start_time
        logging.info("Network evaluation took: {} seconds".format(eval_time))

        if self.network == 'crocubot':
            if self._tensorflow_flags.predict_single_shares:  # Return batch axis to series position
                predict_y = np.swapaxes(predict_y, axis1=1, axis2=2)
            predict_y = np.squeeze(predict_y, axis=1)
        else:
            predict_y = np.expand_dims(predict_y, axis=0)

        means, forecast_covariance = self._data_transformation.inverse_transform_multi_predict_y(predict_y, symbols)
        if not np.isfinite(forecast_covariance).all():
            logging.warning('Forecast covariance contains non-finite values.')

        logging.info('Samples from predicted means: {}'.format(means[0:10]))
        if not np.isfinite(means).all():
            logging.warning('Means found to contain non-finite values.')

        means_pd = pd.Series(np.squeeze(means), index=symbols)

        if self.use_historical_covariance:
            covariance = self.calculate_historical_covariance(predict_data, symbols)
            logging.info('Samples from historical covariance: {}'.format(np.diag(covariance)[0:5]))
            logging.warning('Invoking temporary covariance hack')
            cov_diag = np.diag(covariance) + 1e-4
            covariance = np.diag(cov_diag)
        else:
            covariance = forecast_covariance
            logging.info("Samples from forecast_covariance: {}".format(np.diag(covariance)[0:5]))

        covariance_pd = pd.DataFrame(data=covariance, columns=symbols, index=symbols)

        means_pd, covariance_pd = self.filter_predictions(means_pd, covariance_pd)
        return means_pd, covariance_pd

    def filter_predictions(self, means, covariance):
        """ Remove nans from the series and remove those symbols from the covariance dataframe

        :param pdSeries means:
        :param pdDF covariance:
        :return: pdSeries, pdDF
        """

        means = means.dropna()

        valid_symbols = means.index.tolist()
        covariance = covariance.loc[valid_symbols, valid_symbols]

        return means, covariance

    def filter_nan_samples(self, train_x, train_y):
        """ Remove any sample in zeroth dimension which holds a nan """

        n_samples = train_x.shape[0]
        if n_samples != train_y.shape[0]:
            raise ValueError("x and y sample lengths don't match")

        validity_array = np.zeros(n_samples)
        for i in range(n_samples):
            x_sample = train_x[i, :]
            y_sample = train_y[i, :]
            validity_array[i] = np.isfinite(x_sample).all() and np.isfinite(y_sample).all()

        mask = np.where(validity_array)[0]

        return train_x[mask, :], train_y[mask, :]

    def print_verification_report(self, data, data_name):

        data = data.flatten()
        nans = np.isnan(data).sum()
        infs = np.isinf(data).sum()
        finite_data = data[np.isfinite(data)]
        max_data = np.max(finite_data)
        min_data = np.min(finite_data)
        mean = np.mean(finite_data)
        sigma = np.std(finite_data)

        logging.info("{} Infs, Nans: {}, {}".format(data_name, infs, nans))
        logging.info("{} Min, Max: {}, {}".format(data_name, min_data, max_data))
        logging.info("{} Mean, Sigma: {}, {}".format(data_name, mean, sigma))

        if data_name == 'X_data' and np.abs(mean) > 1e-2:
            logging.warning('Mean of input data is too large')

        if data_name == 'Y_data' and max_data < 1e-2:
            raise ValueError("Y Data not classified")

        return min_data, max_data

    def verify_pricing_data(self, predict_data):
        """ Check for any issues in raw data. """

        close = predict_data['close'].values
        min_price, max_price = self.print_verification_report(close, 'Close')
        if min_price < 1e-3:
            logging.warning("Found an unusually small price: {}".format(min_price))

    def verify_y_data(self, y_data):
        testy = deepcopy(y_data)
        self.print_verification_report(testy, 'Y_data')

    def verify_x_data(self, x_data):
        """Check for nans or crazy numbers.
         """
        testx = deepcopy(x_data).flatten()
        xmin, xmax = self.print_verification_report(testx, 'X_data')

        if xmax > CLIP_VALUE or xmin < -CLIP_VALUE:
            n_clipped_elements = np.sum(CLIP_VALUE < np.abs(testx))
            n_elements = len(testx)
            x_data = np.clip(x_data, a_min=-CLIP_VALUE, a_max=CLIP_VALUE)
            logging.warning("Large inputs detected: clip values exceeding {}".format(CLIP_VALUE))
            logging.info("{} of {} elements were clipped.".format(n_clipped_elements, n_elements))

        return x_data

    def calculate_historical_covariance(self, predict_data, symbols):
        # Call the covariance library

        start_time = timer()
        cov = estimate_covariance(
            data=predict_data,
            n_days=self._covariance_ndays,
            minutes_after_open=self._data_transformation.target_market_minute,
            estimation_method=self._covariance_method,
            exchange_calendar=self._data_transformation.exchange_calendar,
            forecast_interval_in_days=self._data_transformation.target_delta_ndays,
            target_symbols=symbols
        )
        end_time = timer()
        cov_time = end_time - start_time
        logging.info("Historical covariance estimation took:{}".format(cov_time))
        logging.info("Cov shape:{}".format(cov.shape))
        if not np.isfinite(cov).all():
            logging.warning('Covariance matrix computation failed. Contains non-finite values.')
            logging.warning('Problematic data: {}'.format(predict_data))
            logging.warning('Derived covariance: {}'.format(cov))

        return pd.DataFrame(data=cov, columns=symbols, index=symbols)

    def update_configuration(self, config):
        """ Pass on some config entries to data_transformation"""

        config["data_transformation"]["n_classification_bins"] = config["n_classification_bins"]
        config["data_transformation"]["nassets"] = config["nassets"]
        config["data_transformation"]["classify_per_series"] = config["classify_per_series"]
        config["data_transformation"]["normalise_per_series"] = config["normalise_per_series"]

        return config

    def _preprocess_inputs(self, train_x_dict):
        """ Prepare training data to be fed into crocubot. """

        numpy_arrays = []
        for key, value in train_x_dict.items():
            numpy_arrays.append(value)
            logging.info("Appending feature of shape {}".format(value.shape))

        # Currently train_x will have dimensions [features; samples; timesteps; symbols]
        train_x = np.stack(numpy_arrays, axis=0)
        train_x = self.reorder_input_dimensions(train_x)

        # Expand dataset if requested
        if self._tensorflow_flags.predict_single_shares:
            train_x = self.expand_input_data(train_x)

        if self.network == 'dropout':
            logging.info("Reshaping and padding")
            n_samples = train_x.shape[0]

            train_x = np.reshape(train_x, [n_samples, 1, -1, 1])
            n_ticks = train_x.shape[2]
            reps = int(784 / n_ticks)
            if reps > 1:
                train_x = np.tile(train_x, (1, 1, reps, 1))

            pad_elements = 784 - train_x.shape[2]
            train_x = np.pad(train_x, [(0, 0), (0, 0), (0, pad_elements), (0, 0)], mode='reflect')
            train_x = np.reshape(train_x, [n_samples, 28, 28, 1])

        train_x = self.verify_x_data(train_x)

        return train_x.astype(np.float32)  # FIXME: set float32 in data transform, conditional on config file

    def _preprocess_outputs(self, train_y_dict):

        train_y = list(train_y_dict.values())[0]
        train_y = np.swapaxes(train_y, axis1=1, axis2=2)

        if self._tensorflow_flags.predict_single_shares:
            n_feat_y = train_y.shape[2]
            train_y = np.reshape(train_y, [-1, 1, 1, n_feat_y])

        if self.network == 'dropout':
            train_y = np.squeeze(train_y)

        self.verify_y_data(train_y)

        return train_y.astype(np.float32)  # FIXME:set float32 in data transform, conditional on config file

    def gaussianise_series(self, train_x):
        """  Gaussianise each series within each batch - but don't normalise means

        :param nparray train_x: Series in format [batches, features, series]. NB ensure all features
            are of the same kind
        :return: nparray The same data but now each series is gaussianised
        """

        n_batches = train_x.shape[0]

        for batch in range(n_batches):
            train_x[batch, :, :] = gaussianise(train_x[batch, :, :], target_sigma=1.0)

        return train_x

    def reorder_input_dimensions(self, train_x):
        """ Reassign ordering of dimensions.

        :param train_x:  Enters with dimensions  [features; samples; timesteps; series]
        :return: train_x  Now with dimensions  [samples; series ; time; features]
        """

        source = [0, 1, 2, 3]
        destination = [3, 0, 2, 1]
        return np.moveaxis(train_x, source, destination)

    def expand_input_data(self, train_x):
        """Converts to the form where each time series is predicted separately, though companion time series are
            included as auxilliary features
        :param nparray train_x: [samples; series ; time; features]
        :return: nparray The expanded training dataset, still in the format [samples; series ; time; features]
        """

        n_samples = train_x.shape[0]
        n_series = train_x.shape[1]
        n_timesteps = train_x.shape[2]
        n_features = train_x.shape[3]
        n_expanded_samples = n_samples * n_series
        logging.info("Data found to hold {} samples, {} series, {} timesteps, {} features.".format(
                n_samples, n_series, n_timesteps, n_features))

        target_shape = [n_expanded_samples, self._n_input_series, n_timesteps, n_features]
        found_duplicates = False

        if self._n_input_series == 1:
            corr_train_x = train_x.reshape(target_shape)
        else:
            corr_train_x = np.zeros(shape=target_shape)

            for sample in range(n_samples):
                # Series ordering may differ between batches - so we need the correlations for each batch
                data_sample = train_x[sample, :, :, FEATURE_TO_RANK_CORRELATIONS]
                neg_correlation_matrix = - np.corrcoef(data_sample, rowvar=False)  # False since col represents a var
                correlation_indices = neg_correlation_matrix.argsort(axis=1)  # Sort negatives to get descending order

                for series_index in range(n_series):
                    if correlation_indices[series_index, [0]] != series_index:
                        found_duplicates = True
                    sample_number = sample * n_series + series_index
                    for i in range(self._n_input_series):
                        corr_series_index = correlation_indices[series_index, i]
                        corr_train_x[sample_number, :, i] = train_x[sample, :, corr_series_index]

        if found_duplicates:
            logging.warning('Some NaNs or duplicate series were found in the data')

        return corr_train_x

    def initialise_topology(self, n_timesteps):
        """ Set up the network topology based upon the configuration file, and shape of input data. """

        layer_heights = self._configuration['layer_heights']
        layer_widths = self._configuration['layer_widths']
        layer_depths = np.ones(len(layer_heights), dtype=np.int)
        default_layer_types = ['full'] * len(layer_heights)
        layer_types = self._configuration.get('layer_types', default_layer_types)

        # Override input layer to match data
        layer_depths[0] = int(self._n_input_series)
        layer_heights[0] = n_timesteps
        layer_widths[0] = self._n_features

        # Setup convolutional layer configuration
        conv_config = {}
        conv_config["kernel_size"] = self._configuration.get('kernel_size', DEFAULT_CONV_KERNEL_SIZE)
        conv_config["n_kernels"] = self._configuration.get('n_kernels', DEFAULT_N_CONV_FILTERS)
        conv_config["dilation_rates"] = self._configuration.get('dilation_rates', 1)
        conv_config["strides"] = self._configuration.get('strides', 1)

        self._topology = tp.Topology(
            n_series=self._n_input_series,
            n_timesteps=n_timesteps,
            n_forecasts=self._n_forecasts,
            n_classification_bins=self._configuration['n_classification_bins'],
            layer_heights=layer_heights,
            layer_widths=layer_widths,
            layer_depths=layer_depths,
            layer_types=layer_types,
            activation_functions=self._configuration['activation_functions'],
            n_features=self._n_features,
            conv_config=conv_config
        )
