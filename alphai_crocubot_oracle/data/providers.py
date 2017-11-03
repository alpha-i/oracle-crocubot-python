from abc import ABCMeta, abstractmethod
from collections import namedtuple
import logging

import numpy as np
from alphai_data_sources.data_sources import DataSourceGenerator
from alphai_data_sources.generator import BatchGenerator, BatchOptions

from alphai_crocubot_oracle.data.classifier import classify_labels
from alphai_crocubot_oracle.flags import FLAGS

logging.getLogger(__name__).addHandler(logging.NullHandler())

TrainData = namedtuple('TrainData', 'train_x train_y')


class AbstractTrainDataProvider(metaclass=ABCMeta):

    @property
    @abstractmethod
    def n_train_samples(self):
        raise NotImplementedError

    @abstractmethod
    def shuffle_data(self):
        raise NotImplementedError

    @abstractmethod
    def get_batch(self, batch_number, batch_size):
        raise NotImplementedError

    def get_number_of_batches(self, batch_size):
        return int(self.n_train_samples / batch_size) + 1


class TrainDataProvider(AbstractTrainDataProvider):
    def __init__(self, train_x, train_y):
        self._train_data = TrainData(train_x, train_y)

    @property
    def n_train_samples(self):
        return self._train_data.train_x.shape[0]

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
        :param batch_size:
        :return:
        """
        train_x = self._train_data.train_x
        train_y = self._train_data.train_y

        lo_index = batch_number * batch_size
        hi_index = lo_index + batch_size
        batch_x = train_x[lo_index:hi_index, :]
        batch_y = train_y[lo_index:hi_index, :]

        return TrainData(batch_x, batch_y)


class TrainDataProviderForDataSource(AbstractTrainDataProvider):

    def __init__(self, series_name, dtype, n_train_samples, for_training, bin_edges=None):
        self._batch_generator = BatchGenerator()
        self._n_train_samples = n_train_samples
        self._bin_edges = bin_edges
        self._dtype = dtype
        self._for_training = for_training

        data_source_generator = DataSourceGenerator()
        logging.info('Loading data series: {}'.format(series_name))
        self._data_source = data_source_generator.make_data_source(series_name)

    def get_batch(self, batch_number, batch_size):
            batch_generator = BatchGenerator()
            batch_options = BatchOptions(batch_size,
                                         batch_number=batch_number,
                                         train=self._for_training,
                                         dtype=self._dtype
                                         )
            features, labels = batch_generator.get_batch(batch_options, self._data_source)

            if self._bin_edges:
                labels = classify_labels(self._bin_edges, labels)

            return TrainData(features, labels)

    @property
    def n_train_samples(self):
        return self._n_train_samples

    def shuffle_data(self):
        pass

    @property
    def data_source(self):
        return self._data_source
