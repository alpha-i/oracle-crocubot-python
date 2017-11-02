# Interface with quant workflow.
# Trains the network then uses it to make predictions
# Also transforms the data before and after the predictions are made

# A fairly generic interface, in that it can easily applied to other models

import logging
from timeit import default_timer as timer
from copy import deepcopy

import numpy as np
import pandas as pd
import tensorflow as tf

from alphai_crocubot_oracle.data.transformation import FinancialDataTransformation
from alphai_time_series.transform import gaussianise

import alphai_crocubot_oracle.crocubot.train as crocubot
import alphai_crocubot_oracle.crocubot.evaluate as crocubot_eval
from alphai_crocubot_oracle.flags import set_training_flags
import alphai_crocubot_oracle.topology as tp
from alphai_crocubot_oracle import DATETIME_FORMAT_COMPACT
from alphai_crocubot_oracle.covariance import estimate_covariance
from alphai_crocubot_oracle.helpers import TrainFileManager

CLIP_VALUE = 5.0  # Largest number allowed to enter the network
DEFAULT_N_CORRELATED_SERIES = 5
TRAIN_FILE_NAME_TEMPLATE = "{}_train_crocubot"
FLAGS = tf.app.flags.FLAGS

logging.getLogger(__name__).addHandler(logging.NullHandler())


class CrocubotOracle:
    def __init__(self, configuration):
        """
        :param configuration: dictionary containing all the parameters
            data_transformation: Dictionary containing the financial-data-transformation configuration:
                features_dict: Dictionary containing the financial-features configuration, with feature names as keys:
                    order: ['value', 'log-return']
                    normalization: [None, 'robust', 'min_max', 'standard']
                    resample_minutes: resample frequency of feature data_x in minutes.
                    ndays: number of days of feature data_x.
                    start_min_after_market_open: start time of feature data_x in minutes after market open.
                    is_target: boolean to define if this feature is a target (y). The feature is always consider as x.
                exchange_name: name of the exchange to create the market calendar
                prediction_min_after_market_open: prediction time in number of minutes after market open
                target_delta_ndays: days difference between prediction and target
                target_min_after_market_open: target time in number of minutes after market open
            covariance_config:
                covariance_method: The name of the covariance estimation method.
                covariance_ndays: The number of previous days those are needed for the covariance estimate (int).
                use_forecast_covariance: (bool) Whether to use the covariance of the forecast.
                    (If False uses historical data)
            network_config:
                n_series: Number of input time series
                n_features_per_series: Number of inputs associated with each time series
                n_forecasts: Number of outputs to be classified (usually n_series but potentially differs)
                n_classification_bins: Number of bins used for the classification of each forecast
                layer_heights: List of the number of neurons in each layer
                layer_widths: List of the number of neurons in each layer
                activation_functions: list of the activation functions in each layer
                model_save_path: directory where the model is stored
            training_config:
                epochs: The number of epochs in the model training as an integer.
                learning_rate: The learning rate of the model as a float.
                batch_size:  The batch size in training as an integer
                cost_type:  The method for evaluating the loss (default: 'bayes')
                train_path: The path to a folder in which the training data is to be stored.
                resume_training: (bool) whether to load an pre-trained model
            verbose: Is a verbose output required? (bool)
            save_model: If true, save every trained model.
        """

        logging.info('Initialising Crocubot Oracle.')

        configuration = self.update_configuration(configuration)

        self._data_transformation = FinancialDataTransformation(configuration['data_transformation'])
        self._train_path = configuration['train_path']
        self._covariance_method = configuration['covariance_method']
        self._covariance_ndays = configuration['covariance_ndays']

        # FIXME Temporary use default setting for tests to pass
        if 'use_historical_covariance' in configuration:
            self.use_historical_covariance = configuration['use_historical_covariance']
        else:
            self.use_historical_covariance = False

        # FIXME use n_correlated_series configuration.get('n_correlated_series', DEFAULT_N_CORRELATED_SERIES)
        if 'n_correlated_series' in configuration:
            n_correlated_series = configuration['n_correlated_series']
        else:
            n_correlated_series = DEFAULT_N_CORRELATED_SERIES

        self._configuration = configuration
        self._train_file_manager = TrainFileManager(
            self._train_path,
            TRAIN_FILE_NAME_TEMPLATE,
            DATETIME_FORMAT_COMPACT
        )

        self._train_file_manager.ensure_path_exists()
        self._est_cov = None

        # TODO Replace this FLAGS with an actual object
        set_training_flags(configuration)  # Perhaps use separate config dict here?

        if FLAGS.predict_single_shares:
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

        # Topology can either be directly constructed from layers, or build from sequence of parameters
        if self._topology is None:
            features_per_series = train_x.shape[1]
            self.initialise_topology(features_per_series)

        logging.info('Initialised network topology: {}.'.format(self._topology.layers))

        logging.info('Training features of shape: {}.'.format(
            train_x.shape,
        ))
        logging.info('Training labels of shape: {}.'.format(
            train_y.shape,
        ))

        resume_train_path = None

        if FLAGS.resume_training:
            try:
                resume_train_path = self._train_file_manager.latest_train_filename(execution_time)
            except:
                pass
        train_path = self._train_file_manager.new_filename(execution_time)
        data_source = 'financial_stuff'
        start_time = timer()  # TODO replace this with timeit like decorator
        crocubot.train(self._topology, data_source, execution_time, train_x, train_y, save_path=train_path,
                       restore_path=resume_train_path)
        end_time = timer()
        train_time = end_time - start_time
        logging.info("Training took: {} seconds".format(train_time))

    def predict(self, predict_data, execution_time):
        """

        :param dict predict_data: OHLCV data as dictionary of pandas DataFrame
        :param datetime.datetime execution_time: time of execution of prediction

        :return : mean vector (pd.Series) and two covariance matrices (pd.DF)
        """

        if self._topology is None:
            logging.warning('Not ready for prediction - safer to run train first')

        logging.info('Crocubot Oracle prediction on {}.'.format(execution_time))

        self.verify_pricing_data(predict_data)
        latest_train = self._train_file_manager.latest_train_filename(execution_time)
        predict_x, symbols = self._data_transformation.create_predict_data(predict_data)

        logging.info('Predicting mean values.')
        start_time = timer()
        predict_x = self._preprocess_inputs(predict_x)


        if self._topology is None:
            features_per_series = predict_x.shape[1]
            self.initialise_topology(features_per_series)

        # Verify data is the correct shape
        topology_shape = (self._topology.n_features_per_series, self._topology.n_series)

        if predict_x.shape[-2:] != topology_shape:
            raise ValueError('Data shape' + str(predict_x.shape) + " doesnt match network input " + str(topology_shape))

        predict_y = crocubot_eval.eval_neural_net(predict_x, topology=self._topology, save_file=latest_train)

        end_time = timer()
        eval_time = end_time - start_time
        logging.info("Crocubot evaluation took: {} seconds".format(eval_time))

        if FLAGS.predict_single_shares:  # Return batch axis to series position
            predict_y = np.swapaxes(predict_y, axis1=1, axis2=2)

        predict_y = np.squeeze(predict_y, axis=1)
        means, forecast_covariance = self._data_transformation.inverse_transform_multi_predict_y(predict_y, symbols)
        if not np.isfinite(forecast_covariance).all():
            logging.warning('Prediction of forecast covariance failed. Contains non-finite values.')
            logging.warning('forecast_covariance: {}'.format(forecast_covariance))

        if not np.isfinite(means).all():
            logging.warning('Prediction of means failed. Contains non-finite values.')
            logging.warning('Means: {}'.format(means))
        else:
            logging.info('Samples from predicted means: {}'.format(means[0:10]))

        means = pd.Series(np.squeeze(means), index=predict_data['close'].columns)  # FIXME check symbols match cols here

        if self.use_historical_covariance:
            covariance = self.calculate_historical_covariance(predict_data)
            logging.info('Samples from historical covariance: {}'.format(np.diag(covariance)[0:5]))
        else:
            logging.info("Samples from forecast_covariance: {}".format(np.diag(forecast_covariance)[0:5]))
            covariance = pd.DataFrame(data=forecast_covariance, columns=predict_data['close'].columns,
                                      index=predict_data['close'].columns)

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
        logging.info("{} Infs: {}".format(data_name, infs))
        logging.info("{} Nans: {}".format(data_name, nans))
        logging.info("{} Maxs: {}".format(data_name, max_data))
        logging.info("{} Mins: {}".format(data_name, min_data))
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
            n_clipped_elements = np.sum(xmax < np.abs(testx))
            n_elements = len(testx)
            x_data = np.clip(x_data, a_min=-CLIP_VALUE, a_max=CLIP_VALUE)
            logging.warning("Large inputs detected: clip values exceeding {}".format(CLIP_VALUE))
            logging.info("{} of {} elements were clipped.".format(n_clipped_elements, n_elements))

        return x_data

    def calculate_historical_covariance(self, predict_data):

        # Call the covariance library
        logging.info('Estimating historical covariance matrix.')
        start_time = timer()
        cov = estimate_covariance(
            predict_data,
            self._covariance_ndays,
            self._data_transformation.target_market_minute,
            self._covariance_method,
            self._data_transformation.exchange_calendar,
            self._data_transformation.target_delta_ndays
        )
        end_time = timer()
        cov_time = end_time - start_time
        logging.info("Historical covariance estimation took:{}".format(cov_time))
        if not np.isfinite(cov).all():
            logging.warning('Covariance matrix computation failed. Contains non-finite values.')
            logging.warning('Problematic data: {}'.format(predict_data))
            logging.warning('Derived covariance: {}'.format(cov))

        return pd.DataFrame(data=cov, columns=predict_data['close'].columns, index=predict_data['close'].columns)

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

        train_x = np.concatenate(numpy_arrays, axis=1)

        # Expand dataset if requested
        if FLAGS.predict_single_shares:
            train_x = self.expand_input_data(train_x)

        train_x = self.verify_x_data(train_x)

        return train_x.astype(np.float32)  # FIXME: set float32 in data transform, conditional on config file

    def _preprocess_outputs(self, train_y_dict):
        # jut one loop below. a convoluted way of getting the only value out of a dictionary
        for key, value in train_y_dict.items():  # FIXME move this preprocess_outputs
            train_y = value

        train_y = np.swapaxes(train_y, axis1=1, axis2=2)

        if FLAGS.predict_single_shares:
            n_feat_y = train_y.shape[2]
            train_y = np.reshape(train_y, [-1, 1, n_feat_y])  # , order='F'

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

    def expand_input_data(self, train_x):
        """Converts to the form where each time series is predicted separately, though companion time series are
            included as auxilliary features
        :param nparray train_x: The log returns in format [batches, features, series]. Ideally these have been
            Gaussianised already
        :return: nparray The expanded training dataset, still in the format [batches, features, series]
        """

        n_batches = train_x.shape[0]
        n_feat_x = train_x.shape[1]
        n_series = train_x.shape[2]
        n_total_samples = n_batches * n_series

        corr_shape = [n_total_samples, self._n_input_series, n_feat_x]
        corr_train_x = np.zeros(shape=corr_shape)
        found_duplicates = False

        if self._n_input_series == 1:
            train_x = np.swapaxes(train_x, axis1=1, axis2=2)
            corr_train_x = train_x.reshape(corr_shape)
            corr_train_x = np.swapaxes(corr_train_x, axis1=1, axis2=2)
        else:
            raise NotImplementedError('not yet fixed to use multiple correlated series')
            for batch in range(n_batches):
                # Series ordering may differ between batches - so we need the correlations for each batch
                batch_data = train_x[batch, :, :]
                neg_correlation_matrix = - np.corrcoef(batch_data, rowvar=False)  # False since each col represents a var
                correlation_indices = neg_correlation_matrix.argsort(axis=1)  # Sort negative corr to get descending order

                for series_index in range(n_series):
                    if correlation_indices[series_index, [0]] != series_index:
                        found_duplicates = True
                    sample_number = batch * n_series + series_index
                    for i in range(self._n_input_series):
                        corr_series_index = correlation_indices[series_index, i]
                        corr_train_x[sample_number, :, i] = train_x[batch, :, corr_series_index]

        if found_duplicates:
            logging.warning('Some NaNs or duplicate series were found in the data')

        return corr_train_x

    def initialise_topology(self, features_per_series):
        """ Set up the network topology based upon the configuration file, and shape of input data. """

        self._topology = tp.Topology(
            layers=None,
            n_series=self._n_input_series,
            n_features_per_series=features_per_series,
            n_forecasts=self._n_forecasts,
            n_classification_bins=self._configuration['n_classification_bins'],
            layer_heights=self._configuration['layer_heights'],
            layer_widths=self._configuration['layer_widths'],
            activation_functions=self._configuration['activation_functions']
        )
