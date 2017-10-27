import numpy as np

from alphai_crocubot_oracle.data.cleaning import sample_minutes_after_market_open_data_frame

# TODO this method looks like an helper function used only in the covariance module. move it there and make it private

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

# TODO this method is not user anywhere in the main package. Move it
def log_to_simple_returns_conversion(log_returns, covariance_matrix):
    """
    converts the log returns and covariances into simple returns and covariances for use in the portfolio optimisation
    :param np.array log_returns:
    :param np.array covariance_matrix:
    :return tuple: (np.array simple_returns, np.array simple_covariances)
    """

    num_stocks = covariance_matrix.shape[0]
    log_returns = log_returns.reshape([num_stocks, 1])
    diag_covariances = np.diag(covariance_matrix).reshape([num_stocks, 1])

    simple_returns = np.exp(log_returns + 0.5 * diag_covariances) - 1

    tmp_ret_mat = log_returns * np.ones([1, num_stocks]) + log_returns.T * np.ones([num_stocks, 1])
    tmp_cov_mat = diag_covariances.T * np.ones([1, num_stocks]) + diag_covariances * np.ones([num_stocks, 1])
    simple_covariances = np.exp(tmp_ret_mat + 0.5 * tmp_cov_mat) * (np.exp(covariance_matrix) - 1)

    return simple_returns, simple_covariances
