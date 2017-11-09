from copy import deepcopy
import numpy as np
import logging

from alphai_covariance.dynamic_cov import estimate_cov
from alphai_crocubot_oracle.data.cleaning import sample_minutes_after_market_open_data_frame

DEFAULT_N_ESTIMATES = 100
DEFAULT_SPLIT_STEPS = 1
DEFAULT_NUM_REALISATIONS_MULTIPLICATION_FACTOR = 3
MAX_LOG_RETURN_AMPLITUDE = 1


def estimate_covariance(data, n_days, minutes_after_open, estimation_method,
                        exchange_calendar, forecast_interval_in_days, target_symbols=None):
    """
    :param data: OHLCV data
    :param n_days: number of historical days expected for the covariance estimate
    :param minutes_after_open: minutes after the covariance should be calculated
    :param estimation_method: covariance estimation method either NERCOME or Ledoit
    :param exchange_calendar: pandas_market_calendars
    :param forecast_interval_in_days: how many days ahead we should predict?
    :param target_symbols: The symbols we want the covariance for
    :return: The covariance matrix of the data.
    """

    data = returns_minutes_after_market_open_data_frame(data['close'], exchange_calendar, minutes_after_open)

    # Select target symbols
    if target_symbols is not None:
        data = data[target_symbols]

    n_dimensions = data.shape[1]
    sampling_days = np.maximum(n_days, n_dimensions * DEFAULT_NUM_REALISATIONS_MULTIPLICATION_FACTOR)
    data_points = data.values[-sampling_days:, :]
    data_points = clip_data(data_points)

    covariance_matrix, _ = estimate_cov(data_points, method=estimation_method, is_dynamic=False)

    # Rescale amplitude for longer time horizons
    return covariance_matrix * forecast_interval_in_days


def clip_data(x_data):
    """ Reduce outliers to some maximum amplitude. """

    flat_x = deepcopy(x_data).flatten()
    max_data = np.max(np.abs(flat_x))

    if max_data > MAX_LOG_RETURN_AMPLITUDE:
        x_data = np.clip(x_data, a_min=-MAX_LOG_RETURN_AMPLITUDE, a_max=MAX_LOG_RETURN_AMPLITUDE)

        n_clipped_elements = np.sum(MAX_LOG_RETURN_AMPLITUDE < np.abs(flat_x))
        n_elements = len(flat_x)
        logging.warning("Large variance detected: clip values exceeding {}".format(MAX_LOG_RETURN_AMPLITUDE))
        logging.info("{} of {} datapoints were clipped.".format(n_clipped_elements, n_elements))

    return x_data


def returns_minutes_after_market_open_data_frame(data_frame, market_calendar, minutes_after_market_open):
    """
    Daily returns from input dataframe sampled at a specified number of minutes after market opens
    :param data_frame: Dataframe with time as index
    :param market_calendar: pandas_market_calendar
    :param minutes_after_market_open: number of minutes after market opens
    :return: Dataframe of daily returns at specified time after market opens
    """
    sampled_data_frame = \
        sample_minutes_after_market_open_data_frame(data_frame, market_calendar, minutes_after_market_open)
    return np.log(sampled_data_frame.pct_change() + 1).dropna()
