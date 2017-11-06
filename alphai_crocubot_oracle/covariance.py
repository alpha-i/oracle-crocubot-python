import numpy as np
from alphai_covariance.dynamic_cov import estimate_cov

from alphai_crocubot_oracle.data.cleaning import sample_minutes_after_market_open_data_frame

DEFAULT_N_ESTIMATES = 100
DEFAULT_SPLIT_STEPS = 1
DEFAULT_NUM_REALISATIONS_MULTIPLICATION_FACTOR = 3


def estimate_covariance(data, ndays, minutes_after_open, estimation_method,
                        exchange_calendar, forecast_interval_in_days, target_symbols=None):
    """
    :param data: OHLCV data
    :param ndays: number of historical days expected for the covariance estimate
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

    nd = data.shape[1]
    sampling_days = nd * DEFAULT_NUM_REALISATIONS_MULTIPLICATION_FACTOR
    data_points = data.values[-sampling_days:, :]

    covariance_matrix, _ = estimate_cov(data_points, method=estimation_method, is_dynamic=False)

    # Rescale amplitude for longer time horizons
    return covariance_matrix * forecast_interval_in_days


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
