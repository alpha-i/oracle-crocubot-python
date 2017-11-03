import logging
from timeit import default_timer as timer

import numpy as np
import tensorflow as tf
from alphai_data_sources.data_sources import DataSourceGenerator
from alphai_data_sources.generator import BatchOptions, BatchGenerator

from alphai_crocubot_oracle import iotools as io
from alphai_crocubot_oracle.crocubot.helpers import get_tensorboard_log_dir_current_execution
from alphai_crocubot_oracle.data.providers import TrainData, TrainDataProviderForDataSource
from alphai_crocubot_oracle.crocubot.model import CrocuBotModel
from alphai_crocubot_oracle.crocubot.train import _log_topology_parameters_size, FLAGS, _set_cost_operator, \
    _set_training_operator
from alphai_crocubot_oracle.crocubot import PRINT_LOSS_INTERVAL, PRINT_SUMMARY_INTERVAL
from alphai_crocubot_oracle.data.classifier import classify_labels



def train_with_datasource(topology, series_name, execution_time, train_x=None, train_y=None, bin_edges=None,
                          save_path=None,
                          restore_path=None):
    """ Train network on either MNIST or time series data

    FIXME
    :param Topology topology:
    :param str series_name:
    :return: epoch_loss_list
    """

    _log_topology_parameters_size(topology)
    tensorboard_log_dir = get_tensorboard_log_dir_current_execution(execution_time)
    # Start from a clean graph
    tf.reset_default_graph()
    model = CrocuBotModel(topology, FLAGS)
    model.build_layers_variables()

    n_training_samples = FLAGS.n_training_samples_benchmark

    data_provider = TrainDataProviderForDataSource()

    # Placeholders for the inputs and outputs of neural networks
    x = tf.placeholder(FLAGS.d_type, shape=[None, topology.n_features_per_series, topology.n_series], name="x")
    y = tf.placeholder(FLAGS.d_type, name="y")

    global_step = tf.Variable(0, trainable=False, name='global_step')
    n_batches = int(n_training_samples / FLAGS.batch_size) + 1

    cost_operator = _set_cost_operator(model, x, y, n_batches)
    tf.summary.scalar("cost", cost_operator)
    optimize = _set_training_operator(cost_operator, global_step)

    all_summaries = tf.summary.merge_all()

    model_initialiser = tf.global_variables_initializer()

    # TODO set save_path and restore path as required so we can remove the dependency
    if save_path is None:
        save_path = io.build_check_point_filename(series_name, topology)
    saver = tf.train.Saver()

    # Launch the graph
    logging.info("Launching Graph.")
    with tf.Session() as sess:

        if restore_path is not None:
            try:
                logging.info("Attempting to load model from {}".format(restore_path))
                saver.restore(sess, restore_path)
                logging.info("Model restored.")
                n_epochs = FLAGS.n_retrain_epochs
            except:
                logging.warning("Restore file not recovered. Training from scratch")
                n_epochs = FLAGS.n_epochs
                sess.run(model_initialiser)
        else:
            logging.info("Initialising new model.")
            n_epochs = FLAGS.n_epochs
            sess.run(model_initialiser)

        summary_writer = tf.summary.FileWriter(tensorboard_log_dir)

        epoch_loss_list = []
        for epoch in range(n_epochs):
            # TODO replace this timer with logtime like decorator
            epoch_loss = 0.
            start_time = timer()

            for batch_number in range(n_batches):  # The randomly sampled weights are fixed within single batch

                batch_data = data_provider.get_batch(batch_number, FLAGS.batch_size)
                batch_x = batch_data.train_x
                batch_y = batch_data.train_y

                if batch_number == 0 and epoch == 0:
                    logging.info("Training {} batches of size {} and {}"
                                 .format(n_batches, batch_x.shape, batch_y.shape))

                _, batch_loss, summary_results = sess.run([optimize, cost_operator, all_summaries],
                                                          feed_dict={x: batch_x, y: batch_y})
                epoch_loss += batch_loss

                if epoch * batch_number % PRINT_SUMMARY_INTERVAL:
                    summary_index = epoch * n_batches + batch_number
                    summary_writer.add_summary(summary_results, summary_index)

            time_epoch = timer() - start_time

            if epoch_loss != epoch_loss:
                raise ValueError("Found nan value for epoch loss.")

            epoch_loss_list.append(epoch_loss)

            if (epoch % PRINT_LOSS_INTERVAL) == 0:
                msg = "Epoch {} of {} ... Loss: {:.2e}. in {:.2f} seconds.".format(epoch + 1, n_epochs, epoch_loss,
                                                                                   time_epoch)
                logging.info(msg)
                # accuracy_test

        out_path = saver.save(sess, save_path)
        logging.info("Model saved in file:{}".format(out_path))

    return epoch_loss_list


# def get_batch_from_generator(batch_options, data_source, bin_edges=None):
#     batch_generator = BatchGenerator()
#     features, labels = batch_generator.get_batch(batch_options, data_source)
#
#     if bin_edges is not None:
#         labels = classify_labels(bin_edges, labels)
#
#     return features, labels
