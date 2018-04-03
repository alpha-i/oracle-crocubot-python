import os

from alphai_crocubot_oracle import DATETIME_FORMAT_COMPACT


class TensorflowPath:
    def __init__(self, session_save_path, model_restore_path=None):
        self._session_save_path = session_save_path
        self._model_restore_path = model_restore_path

    def can_restore_model(self):
        return isinstance(self._model_restore_path, str)

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
