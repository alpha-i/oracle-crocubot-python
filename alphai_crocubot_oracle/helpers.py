# Used by oracle.py to keep track of the names of the save files

import os
import glob
import time
import logging

from alphai_crocubot_oracle import DATETIME_FORMAT_COMPACT

logging.getLogger(__name__).addHandler(logging.NullHandler())


class TrainFileManager:
    """
    This class manage read and retrieval of the calibration file for the training.
    """

    TIMESTAMP_CHARS_NUMBER = 14

    def __init__(self, path, file_name_template, datetime_format=DATETIME_FORMAT_COMPACT):
        """

        :param str path: training file directory
        :param str file_name_template: training filename template
        :param str datetime_format: the format string of the datetime used for creating the file
        """
        self._path = path
        self._file_name_template = file_name_template
        self._datetime_format = datetime_format

    def new_filename(self, execution_time):
        """
        Return new filename given the execution_time
        :param datetime.datetime execution_time:
        :return string:
        """
        return os.path.join(
            self._path,
            self._file_name_template.format(execution_time.strftime(self._datetime_format))
        )

    def ensure_path_exists(self):
        """
        Creates path if it doesn't exists
        :return:
        """
        if not os.path.exists(self._path):
            os.makedirs(self._path)

    def latest_train_filename(self, execution_time):
        """
        Given a timestamp, it returns the latest useful calibration file, if exists
        :param datetime.datetime execution_time:

        :return string:
        """
        calibration_files = glob.glob1(self._path, self._file_name_template.format("*") + "*")
        execution_timestamp = int(execution_time.strftime(self._datetime_format))
        latest_calibration = None
        for calibration_file_name in sorted(calibration_files):
            calibration_timestamp = int(calibration_file_name[:self.TIMESTAMP_CHARS_NUMBER])
            if execution_timestamp >= calibration_timestamp:
                latest_calibration = calibration_file_name.split('.')[0]
            else:
                break

        if not latest_calibration:
            raise ValueError("No calibration found before {}".format(execution_timestamp))

        return os.path.join(self._path, latest_calibration)


def logtime(message=None):
    def wrap(method):
        def wrapped_f(*args, **kw):
            ts = time.time()
            result = method(*args, **kw)
            te = time.time()
            duration = te - ts
            msg = message if message else method.__name__
            logging.info("{}: Execution time: {} seconds".format(msg, duration))
            return result

        return wrapped_f

    return wrap


def printtime(message=None):
    def wrap(method):
        def wrapped_f(*args, **kw):
            ts = time.time()
            result = method(*args, **kw)
            te = time.time()
            duration = te - ts
            msg = message if message else method.__name__
            print("{}: Execution time: {} seconds".format(msg, duration))
            return result

        return wrapped_f

    return wrap


def execute_and_get_duration(method, *args, **kw):
    ts = time.time()
    result = method(*args, **kw)
    te = time.time()
    duration = te - ts
    return duration, result
