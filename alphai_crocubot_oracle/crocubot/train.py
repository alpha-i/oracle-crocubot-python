# Trains the network
# Used by oracle.py

import logging
from timeit import default_timer as timer
import os
import tensorflow as tf
import numpy as np

import alphai_crocubot_oracle.bayesian_cost as cost
from alphai_crocubot_oracle.crocubot.model import CrocuBotModel, Estimator
import alphai_crocubot_oracle.iotools as io
from alphai_crocubot_oracle import DATETIME_FORMAT_COMPACT

from alphai_data_sources.data_sources import DataSourceGenerator
from alphai_data_sources.generator import BatchOptions

FLAGS = tf.app.flags.FLAGS
PRINT_LOSS_INTERVAL = 20
PRINT_SUMMARY_INTERVAL = 5
MAX_GRADIENT = 7.0
PRINT_KERNEL = True


# TODO encapsulate the parameters in a ParameterObject
# TODO remove FLAGS usage
def train(topology, series_name, execution_time, train_x=None, train_y=None, bin_edges=None, save_path=None,
          restore_path=None):
    """ Train network on either MNIST or time series data

    FIXME
    :param Topology topology:
    :param str series_name:
    :return: epoch_loss_list
    """

    _verify_topology(topology)
    tensorboard_log_dir = get_tensorboard_log_dir_current_execution(execution_time)
    # Start from a clean graph
    tf.reset_default_graph()
    model = CrocuBotModel(topology, FLAGS)
    model.build_layers_variables()

    if train_x is None:
        use_data_loader = True
        n_training_samples = FLAGS.n_training_samples_benchmark
    else:
        use_data_loader = False
        n_training_samples = train_x.shape[0]

    if use_data_loader:
        data_source_generator = DataSourceGenerator()
        batch_options = BatchOptions(FLAGS.batch_size, batch_number=0, train=True, dtype=FLAGS.d_type)
        logging.info('Loading data series: {}'.format(series_name))
        data_source = data_source_generator.make_data_source(series_name)

    # Placeholders for the inputs and outputs of neural networks

    x_shape = [None, topology.n_series, topology.n_timesteps, topology.n_features]
    x = tf.placeholder(FLAGS.d_type, shape=x_shape, name="x")
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
        save_path = io.load_file_name(series_name, topology)
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
            if train_x is not None:
                train_x, train_y = shuffle_training_data(train_x, train_y)

            # TODO replace this timer with timeit like decorator
            epoch_loss = 0.
            start_time = timer()

            for batch_number in range(n_batches):  # The randomly sampled weights are fixed within single batch

                # TODO implement a smarter version of DatasourceGenerator to remove the iotools dependency
                # TODO and replace alphai_crocubot_oracle.iotools.load_batch
                if use_data_loader:
                    batch_options.batch_number = batch_number
                    batch_x, batch_y = io.load_batch(batch_options, data_source, bin_edges=bin_edges)
                else:
                    batch_x, batch_y = extract_batch(train_x, train_y, batch_number)

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

                # Print convolutional kernel
                if PRINT_KERNEL and FLAGS.use_convolution:
                    gr = tf.get_default_graph()
                    conv1_kernel_val = gr.get_tensor_by_name('conv3d0/kernel:0').eval()
                    conv1_bias_val = gr.get_tensor_by_name('conv3d0/bias:0').eval()
                    logging.info("Kernel values: {}".format(conv1_kernel_val.flatten()))
                    logging.info("Kernel bias: {}".format(conv1_bias_val))

        out_path = saver.save(sess, save_path)
        logging.info("Model saved in file:{}".format(out_path))

    return epoch_loss_list


def extract_batch(x, y, batch_number):
    """ Returns batch of features and labels from the full data set x and y

    :param nparray x: Full set of training features
    :param nparray y: Full set of training labels
    :param int batch_number: Which batch
    :return:
    """
    lo_index = batch_number * FLAGS.batch_size
    hi_index = lo_index + FLAGS.batch_size
    batch_x = x[lo_index:hi_index, :]
    batch_y = y[lo_index:hi_index, :]

    return batch_x, batch_y


# TODO remove FLAGS
def _set_cost_operator(crocubot_model, x, labels, n_batches):
    """
    Set the cost operator

    :param CrocubotModel crocubot_model:
    :param data x:
    :param labels:
    :return:
    """

    cost_object = cost.BayesianCost(crocubot_model,
                                    FLAGS.double_gaussian_weights_prior,
                                    FLAGS.wide_prior_std,
                                    FLAGS.narrow_prior_std,
                                    FLAGS.spike_slab_weighting,
                                    n_batches
                                    )

    estimator = Estimator(crocubot_model, FLAGS)

    if FLAGS.cost_type == 'bayes':
        log_predictions = estimator.average_multiple_passes(x, FLAGS.n_train_passes)
        operator = cost_object.get_bayesian_cost(log_predictions, labels)
    elif FLAGS.cost_type == 'bbalpha':
        operator = cost_object.get_hellinger_cost(x, labels, FLAGS.n_train_passes, estimator)
    elif FLAGS.cost_type == 'softmax':
        log_predictions = estimator.average_multiple_passes(x, FLAGS.n_train_passes)
        operator = tf.nn.softmax_cross_entropy_with_logits(logits=log_predictions, labels=labels)
    else:
        raise NotImplementedError

    return tf.reduce_mean(operator)


def _verify_topology(topology):
    """Check topology is sensible """

    logging.info("Requested topology: {}".format(topology.layers))

    if topology.n_parameters > 1e7:
        logging.warning("Ambitious number of parameters: {}".format(topology.n_parameters))
    else:
        logging.info("Number of parameters: {}".format(topology.n_parameters))


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


# TODO remove the usage of FLAGS. Create a Provider for training_operator
def _set_training_operator(cost_operator, global_step):
    """ Define the algorithm for updating the trainable variables. """

    if FLAGS.optimisation_method == 'Adam':
        optimizer = tf.train.AdamOptimizer(FLAGS.learning_rate)
        gradients, variables = zip(*optimizer.compute_gradients(cost_operator))
        gradients, _ = tf.clip_by_global_norm(gradients, MAX_GRADIENT)
        optimize = optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)
    elif FLAGS.optimisation_method == 'GDO':
        optimizer = tf.train.GradientDescentOptimizer(FLAGS.learning_rate)
        grads_and_vars = optimizer.compute_gradients(cost_operator)
        clipped_grads_and_vars = [(tf.clip_by_value(g, -MAX_GRADIENT, MAX_GRADIENT), v) for g, v in grads_and_vars]
        optimize = optimizer.apply_gradients(clipped_grads_and_vars)
    else:
        raise NotImplementedError("Unknown optimisation method: ", FLAGS.optimisation_method)

    return optimize


def shuffle_training_data(train_x, train_y):
    """ Reorder the numpy arrays in a random but consistent manner """

    rng_state = np.random.get_state()
    np.random.shuffle(train_x)
    np.random.set_state(rng_state)
    np.random.shuffle(train_y)

    return train_x, train_y
