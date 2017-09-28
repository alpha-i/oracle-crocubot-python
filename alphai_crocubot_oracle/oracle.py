# Interface with quant workflow.
# Trains the network then uses it to make predictions
# Also transforms the data before and after the predictions are made

# A fairly generic interface, in that it can easily applied to other models

import logging
from timeit import default_timer as timer

import numpy as np
import pandas as pd
import tensorflow as tf

from alphai_finance.data.transformation import FinancialDataTransformation
from alphai_time_series.transform import gaussianise

import alphai_crocubot_oracle.crocubot.train as crocubot
import alphai_crocubot_oracle.crocubot.evaluate as crocubot_eval
from alphai_crocubot_oracle.flags import set_training_flags
import alphai_crocubot_oracle.topology as tp
from alphai_crocubot_oracle.constants import DATETIME_FORMAT_COMPACT
from alphai_crocubot_oracle.covariance import estimate_covariance
from alphai_crocubot_oracle.helpers import TrainFileManager

DEFAULT_N_CORRELATED_SERIES = 5
TRAIN_FILE_NAME_TEMPLATE = "{}_train_crocubot"
FLAGS = tf.app.flags.FLAGS
MAX_INPUT_AMPLITUDE = 6

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
                prediction_frequency_ndays: frequency of the prediction in days
                prediction_min_after_market_open: prediction time in number of minutes after market open
                target_delta_ndays: days difference between prediction and target
                target_min_after_market_open: target time in number of minutes after market open
            covariance_config:
                covariance_method: The name of the covariance estimation method.
                covariance_ndays: The number of previous days those are needed for the covariance estimate (int).
                use_forecast_covariance: (bool) Whether to use the covariance of the forecast. (If False uses historical data)
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

        self._data_transformation = FinancialDataTransformation(configuration['data_transformation'])
        self._train_path = configuration['train_path']
        self._covariance_method = configuration['covariance_method']
        self._covariance_ndays = configuration['covariance_ndays']

        self._train_file_manager = TrainFileManager(
            self._train_path,
            TRAIN_FILE_NAME_TEMPLATE,
            DATETIME_FORMAT_COMPACT
        )

        self._train_file_manager.ensure_path_exists()
        self._est_cov = None

        set_training_flags(configuration)  # Perhaps use separate config dict here?

        if FLAGS.predict_single_shares:
            self._n_input_series = int(np.minimum(DEFAULT_N_CORRELATED_SERIES, configuration['n_series']))
            self._n_forecasts = 1
        else:
            self._n_input_series = configuration['n_series']
            self._n_forecasts = configuration['n_forecasts']

        # Topology can either be directly constructed from layers, or build from sequence of parameters
        self._topology = tp.Topology(
            layers=None,
            n_series=self._n_input_series,
            n_features_per_series=configuration['n_features_per_series'],
            n_forecasts=self._n_forecasts,
            n_classification_bins=configuration['n_classification_bins'],
            layer_heights=configuration['layer_heights'],
            layer_widths=configuration['layer_widths'],
            activation_functions=configuration['activation_functions']
        )

        logging.info('Initialised network topology: {}.'.format(self._topology.layers))

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

        train_x, train_y = self._data_transformation.create_train_data(train_data, historical_universes)

        logging.info("Preprocessing training data")
        train_x = self._preprocess_inputs(train_x)
        train_y = self._preprocess_outputs(train_y)

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
        start_time = timer()  # FIXME we should find a way to make some function 'temporizable' with a python decorator
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
        latest_train = self._train_file_manager.latest_train_filename(execution_time)

        logging.info('Crocubot Oracle prediction on {}.'.format(
            execution_time,
        ))

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
            raise ValueError('Covariance matrix computation failed. Contains non-finite values.')
        # Convert the array into a dataframe
        # historical_covariance = pd.DataFrame(data=cov, columns=predict_data['close'].columns, index=predict_data['close'].columns)

        predict_x = self._data_transformation.create_predict_data(predict_data)
        predict_x = np.expand_dims(predict_x, axis=0)  # Effective batch size of 1

        logging.info('Predicting mean values.')
        start_time = timer()

        predict_x = self._preprocess_inputs(predict_x)

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
        means, forecast_covariance = self._data_transformation.inverse_transform_multi_predict_y(predict_y)

        if not np.isfinite(forecast_covariance).all():
            raise ValueError('Prediction of forecast covariance failed. Contains non-finite values.')

        logging.info("Samples from forecast_covariance: {}".format(np.diag(forecast_covariance)[0:5]))

        forecast_covariance = pd.DataFrame(data=forecast_covariance, columns=predict_data['close'].columns,
                                           index=predict_data['close'].columns)

        if not np.isfinite(means).all():
            raise ValueError('Prediction of means failed. Contains non-finite values.')

        means = pd.Series(np.squeeze(means), index=predict_data['close'].columns)

        # return means, historical_covariance, forecast_covariance
        return means, forecast_covariance

    def _preprocess_inputs(self, train_x):
        """ Prepare training data to be fed into crocubot. """

        train_x = np.squeeze(train_x, axis=3)

        # Gaussianise & Normalise of inputs (not necessary for outputs)
        # train_x = self.gaussianise_series(train_x)
        train_x = 300 * train_x   # FIXME: hack 15min universal normalisation issue for now until alpha-i finance is updated
        train_x = np.clip(train_x, -MAX_INPUT_AMPLITUDE, MAX_INPUT_AMPLITUDE)  # Prevent extreme outliers from entering network

        logging.info("Sample features from rescaled train_x: {}".format(train_x[0, 0:10, 0]))
        logging.info("x Shape: {}".format(train_x.shape))

        # Expand dataset if requested
        if FLAGS.predict_single_shares:
            train_x = self.expand_input_data(train_x)

        return train_x.astype(np.float32)  # FIXME: set float32 in data transform, conditional on config file

    def _preprocess_outputs(self, train_y):

        if FLAGS.predict_single_shares:
            n_feat_y = train_y.shape[2]
            train_y = np.reshape(train_y, [-1, 1, n_feat_y])

        return train_y.astype(np.float32)  # FIXME:set float32 in data transform, conditional on config file

    def gaussianise_series(self, train_x):
        """  Gaussianise each series within each batch - but don't normalise means

        :param nparray train_x: Series in format [batches, features, series]. NB ensure all features are of the same kind
        :return: nparray The same data but now each series is gaussianised
        """

        n_batches = train_x.shape[0]

        for batch in range(n_batches):
            train_x[batch, :, :] = gaussianise(train_x[batch, :, :], target_sigma=1.0)

        return train_x

    def expand_input_data(self, train_x):
        """Converts to the form where each time series is predicted separately, though companion time series are included as auxilliary features
        :param nparray train_x: The log returns in format [batches, features, series]. Ideally these have been gaussianised already
        :return: nparray The expanded training dataset, still in the format [batches, features, series]
        """

        n_batches = train_x.shape[0]
        n_feat_x = train_x.shape[1]
        n_series = train_x.shape[2]
        n_total_samples = n_batches * n_series

        corr_train_x = np.zeros(shape=[n_total_samples, n_feat_x, self._n_input_series])
        found_duplicates = False

        for batch in range(n_batches):
            # Series ordering may differ between batches - so we need the correlations for each batch
            batch_data = train_x[batch, :, :]
            neg_correlation_matrix = - np.corrcoef(batch_data, rowvar=False)  # False since each col represents a variable
            correlation_indices = neg_correlation_matrix.argsort(axis=1)  # Sort negative corr to get descending order

            for series_index in range(n_series):
                if correlation_indices[series_index, [0]] != series_index:
                    found_duplicates = True
                sample_number = batch * n_series + series_index
                for i in range(self._n_input_series):
                    corr_series_index = correlation_indices[series_index, i]
                    corr_train_x[sample_number, :, i] = train_x[batch, :, corr_series_index]

        if found_duplicates:
            logging.warning('A series should always be most correlated with itself! Probably duplicate series in the data')

        return corr_train_x
