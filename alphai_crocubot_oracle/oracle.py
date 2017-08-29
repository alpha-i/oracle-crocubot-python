# Interface with quant workflow.
# Trains the network then uses it to make predictions
# Also transforms the data before and after the predictions are made

# A fairly generic interface, in that it can easily applied to other models
# Keras/pb specific calls are commented out to avoid introducing extra dependencies

import logging

import numpy as np
import pandas as pd
import tensorflow as tf

from alphai_finance.data.transformation import FinancialDataTransformation

import alphai_crocubot_oracle.classifier as cl
import alphai_crocubot_oracle.crocubot.train as crocubot
import alphai_crocubot_oracle.crocubot.evaluate as crocubot_eval
import alphai_crocubot_oracle.flags as fl
import alphai_crocubot_oracle.topology as tp
from alphai_crocubot_oracle.constants import DATETIME_FORMAT_COMPACT
from alphai_crocubot_oracle.covariance import estimate_covariance
from alphai_crocubot_oracle.helpers import TrainFileManager

TRAIN_FILE_NAME_TEMPLATE = "{}_train_crocubot.hd5"
FLAGS = tf.app.flags.FLAGS

logging.getLogger(__name__).addHandler(logging.NullHandler())


class MvpOracle:
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

        logging.info('Initialising MVP Oracle.')

        self._data_transformation = FinancialDataTransformation(configuration['data_transformation'])
        self._train_path = configuration['train_path']
        self._covariance_method = configuration['covariance_method']
        self._covariance_ndays = configuration['covariance_ndays']
        self._epochs = configuration['epochs']
        self._learning_rate = configuration['learning_rate']
        self._verbose = configuration['verbose']
        self._batch_size = configuration['batch_size']
        self._n_classification_bins = configuration['n_classification_bins']
        self._drop_out = configuration['drop_out']
        self._l2 = configuration['l2']
        self._n_hidden = configuration['n_hidden']
        self._save_model = configuration['save_model']
        self._ml_library = configuration['ml_library']

        self._train_file_manager = TrainFileManager(
            self._train_path,
            TRAIN_FILE_NAME_TEMPLATE,
            DATETIME_FORMAT_COMPACT
        )

        self._train_file_manager.ensure_path_exists()
        self._ml_model = None
        self._est_cov = None
        self._current_train = None

        assert self._ml_library in ['keras', 'TF'], 'ml_library needs to be in [keras, TF]'

        if self._ml_library == 'TF':
            fl.set_training_flags(configuration)  # Perhaps use separate config dict here?
            # Topology can either be directly constructed from layers, or build from sequence of parameters
            self._topology = tp.Topology(layers=None, n_series=configuration['n_series'], n_features_per_series=configuration['n_features_per_series'], n_forecasts=configuration['n_forecasts'],
                                         n_classification_bins=configuration['n_classification_bins'], layer_heights=configuration['layer_heights'],
                                         layer_widths=configuration['layer_widths'], activation_functions=configuration['activation_functions'])

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

        assert train_y.shape[1] == 1
        train_x = train_x.swapaxes(2, 3).swapaxes(1, 3)  # FIXME probably clearer to use np.tranpose
        train_y = train_y.reshape(train_y.shape[0], train_y.shape[2])

        # if self._ml_library == 'keras':
        # model, history = pb.train_model(train_x, train_y, epochs=self._epochs, lr=self._learning_rate,
        #                                 verbose=self._verbose, batch_size=self._batch_size, do=self._drop_out,
        #                                 l2=self._l2, n_hidden=self._n_hidden)
        # self._ml_model = model
        #
        # if self._save_model:
        #     train_path = self._train_file_manager.new_filename(execution_time)
        #     self._current_train = train_path
        #     model.save(train_path)

        if self._ml_library == 'TF':

            train_x = np.squeeze(train_x, axis=3)

            # Classify the training labels - FIXME: These two lines will be moved inside _data_transformation
            bin_distribution = cl.make_template_distribution(train_y, self._n_classification_bins)
            train_y = cl.classify_labels(bin_distribution["bin_edges"], train_y)

            train_path = self._train_file_manager.new_filename(execution_time)
            data_source = 'financial_stuff'
            crocubot.train(self._topology, data_source, FLAGS, train_x, train_y, save_path=train_path)

            self._current_train = train_path
            self._bin_distribution = bin_distribution

        else:
            raise NotImplementedError

    def predict(self, predict_data, execution_time):
        """

        :param dict predict_data: OHLCV data as dictionary of pandas DataFrame
        :param datetime.datetime execution_time: time of execution of prediction

        :return : mean vector (pd.Series) and two covariance matrices (pd.DF)
        """
        if self._save_model:
            self._load_latest_train(execution_time)

        logging.info('MVP Oracle prediction on {}.'.format(
            execution_time,
        ))
        if self._ml_library == 'keras' and self._ml_model is None:
            raise ValueError('No trained ML model available for prediction.')

        # Call the covariance library
        logging.info('Estimating historical covariance matrix.')
        cov = estimate_covariance(
            predict_data,
            self._covariance_ndays,
            self._data_transformation.target_min_after_market_open,
            self._covariance_method,
            self._data_transformation.exchange_calendar,
            self._data_transformation.target_delta_ndays
        )
        if not np.isfinite(cov).all():
            raise ValueError('Covariance matrix computation failed. Contains non-finite values.')
        # Convert the array into a dataframe
        historical_covariance = pd.DataFrame(data=cov, columns=predict_data['close'].columns, index=predict_data['close'].columns)

        predict_x = self._data_transformation.create_predict_data(predict_data)

        logging.info('Predicting mean values.')
        if self._ml_library == 'keras':
            means = self._ml_model.predict(predict_x)
            forecast_covariance = None
        elif self._ml_library == 'TF':
            binned_forecasts = crocubot_eval.eval_neural_net(predict_x, topology=self._topology, save_file=self._current_train)
            # FIXME: The de-classification operation will be moved inside _data_transformation
            means, forecast_cov = crocubot_eval.forecast_means_and_variance(binned_forecasts, self._bin_distribution)
            if not np.isfinite(forecast_cov).all():
                raise ValueError('Prediction of forecast covariance failed. Contains non-finite values.')

            forecast_covariance = pd.DataFrame(data=forecast_cov, columns=predict_data['close'].columns,
                                               index=predict_data['close'].columns)
        else:
            raise NotImplementedError

        if not np.isfinite(means).all():
            raise ValueError('Prediction of means failed. Contains non-finite values.')

        means = pd.Series(np.squeeze(means), index=predict_data['close'].columns)
        return means, historical_covariance, forecast_covariance

    def _load_latest_train(self, execution_time):

        latest_train = self._train_file_manager.latest_train_filename(
            execution_time
        )

        if latest_train != self._current_train:
            if self._ml_library == 'keras':
                # self._ml_model = load_model(
                #     os.path.join(
                #         self._train_path,
                #         latest_train
                #     )
                # )
                self._current_train = latest_train
            elif self._ml_library == 'TF':  # Don't load model, just location of save file
                self._current_train = latest_train
                raise Warning("Probabiy shouldnt have got here. But lets see what happens...")
            else:
                raise NotImplementedError
