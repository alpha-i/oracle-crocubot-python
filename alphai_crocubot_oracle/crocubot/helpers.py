import os
import numpy as np
from collections import namedtuple
from alphai_crocubot_oracle import DATETIME_FORMAT_COMPACT
from alphai_crocubot_oracle.flags import FLAGS


def get_tensorboard_log_dir_current_execution(execution_time):
    """
    A function that creates unique tensorboard directory given a set of hyper parameters and execution time.

    FIXME I have removed priting of hyper parameters from the log for now.
    The problem is that at them moment {learning_rate, batch_size} are the only hyper parameters.
    In general this is not true. We will have more. We need to find an elegant way of creating a
    unique id for the execution.

    :param learning_rate: Learning rate for the training
    :param batch_size: batch size of the traning
    :param tensorboard_log_path: Root path of the tensorboard logs
    :param execution_time: The execution time for which a unique directory is to be created.
    :return: A unique directory path inside tensorboard path.
    """
    # TODO remove make tensorflow_log_path as required parameter intead of using FLAGS
    hyper_param_string = "lr={}_bs={}".format(FLAGS.learning_rate, FLAGS.batch_size)
    execution_string = execution_time.strftime(DATETIME_FORMAT_COMPACT)
    return os.path.join(FLAGS.tensorboard_log_path, hyper_param_string, execution_string)


class TensorflowPath:
    def __init__(self, session_save_path, model_restore_path=None):
        self._session_save_path = session_save_path
        self._model_restore_path = model_restore_path

    def can_restore_model(self):
        return self._model_restore_path and os.path.isfile(self._model_restore_path)

    @property
    def session_save_path(self):
        return self._session_save_path

    @property
    def model_restore_path(self):
        return self._model_restore_path


class TensorboardOptions:
    def __init__(self, tensorboard_log_path, learning_rate, batch_size, execution_time):
        self._tensorboard_log_path = tensorboard_log_path
        self._learning_rate = learning_rate
        self._batch_size = batch_size
        self.execution_time = execution_time

    def get_log_dir(self):
        """
          A function that creates unique tensorboard directory given a set of hyper parameters and execution time.

          FIXME I have removed priting of hyper parameters from the log for now.
          The problem is that at them moment {learning_rate, batch_size} are the only hyper parameters.
          In general this is not true. We will have more. We need to find an elegant way of creating a
          unique id for the execution.
        """

        hyper_param_string = "lr={}_bs={}".format(self._learning_rate, self._batch_size)
        execution_string = self.execution_time.strftime(DATETIME_FORMAT_COMPACT)

        return os.path.join(self._tensorboard_log_path, hyper_param_string, execution_string)


TrainData = namedtuple('TrainData', 'train_x train_y')


class TrainDataProvider:
    def __init__(self, train_x, train_y):
        self._train_data = TrainData(train_x, train_y)

    @property
    def train_data(self):
        return self._train_data

    @property
    def number_of_training_samples(self):
        return self.train_data.train_x.shape[0]

    def get_number_of_batches(self, batch_size):
        return int(self.number_of_training_samples / batch_size) + 1

    def shuffle_data(self):
        """ Reorder the numpy arrays in a random but consistent manner """

        train_x = self._train_data.train_x
        train_y = self._train_data.train_y

        rng_state = np.random.get_state()
        np.random.shuffle(train_x)
        np.random.set_state(rng_state)
        np.random.shuffle(train_y)

        self._train_data = TrainData(train_x, train_y)

    def get_batch(self, batch_number, batch_size):
        """ Returns batch of features and labels from the full data set x and y

        :param nparray x: Full set of training features
        :param nparray y: Full set of training labels
        :param int batch_number: Which batch
        :return:
        """
        train_x = self._train_data.train_x
        train_y = self._train_data.train_y

        lo_index = batch_number * batch_size
        hi_index = lo_index + batch_size
        batch_x = train_x[lo_index:hi_index, :]
        batch_y = train_y[lo_index:hi_index, :]

        return TrainData(batch_x, batch_y)
