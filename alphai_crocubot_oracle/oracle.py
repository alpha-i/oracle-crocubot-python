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

import alphai_crocubot_oracle.crocubot.train as crocubot
import alphai_crocubot_oracle.crocubot.evaluate as crocubot_eval
from alphai_crocubot_oracle.flags import set_training_flags
import alphai_crocubot_oracle.topology as tp
from alphai_crocubot_oracle.constants import DATETIME_FORMAT_COMPACT
from alphai_crocubot_oracle.covariance import estimate_covariance
from alphai_crocubot_oracle.helpers import TrainFileManager

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

        # Topology can either be directly constructed from layers, or build from sequence of parameters
        self._topology = tp.Topology(
            layers=None,
            n_series=configuration['n_series'],
            n_features_per_series=configuration['n_features_per_series'],
            n_forecasts=configuration['n_forecasts'],
            n_classification_bins=configuration['n_classification_bins'],
            layer_heights=configuration['layer_heights'],
            layer_widths=configuration['layer_widths'],
            activation_functions=configuration['activation_functions']
        )

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

        train_x = np.squeeze(train_x, axis=3).astype(np.float32)  # FIXME: prob do this in data transform, conditional on config file
        train_y = train_y.astype(np.float32)  # FIXME: prob do this in data transform, conditional on config file

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
        start_time = timer()
        crocubot.train(self._topology, data_source, train_x, train_y, save_path=train_path, restore_path=resume_train_path)
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

        logging.info('Predicting mean values.')

        # FIXME: temporary fix, to be added to data transform
        predict_x = np.squeeze(predict_x, axis=2).astype(np.float32)

        # Verify data is the correct shape
        topology_shape = (self._topology.n_features_per_series, self._topology.n_series)
        if predict_x.shape != topology_shape:
            raise ValueError('Data shape' + str(predict_x.shape) + " doesnt match network input " + str(topology_shape))

        start_time = timer()
        predict_y = crocubot_eval.eval_neural_net(predict_x.reshape((1,) + predict_x.shape),
                                                  topology=self._topology, save_file=latest_train)
        end_time = timer()
        eval_time = end_time - start_time
        logging.info("Crocubot evaluation took: {} seconds".format(eval_time))

        predict_y = np.squeeze(predict_y, axis=1)
        means, forecast_covariance = self._data_transformation.inverse_transform_multi_predict_y(predict_y)

        if not np.isfinite(forecast_covariance).all():
            raise ValueError('Prediction of forecast covariance failed. Contains non-finite values.')

        forecast_covariance = pd.DataFrame(data=forecast_covariance, columns=predict_data['close'].columns,
                                           index=predict_data['close'].columns)

        if not np.isfinite(means).all():
            raise ValueError('Prediction of means failed. Contains non-finite values.')

        means = pd.Series(np.squeeze(means), index=predict_data['close'].columns)
        # return means, historical_covariance, forecast_covariance
        return means, forecast_covariance
