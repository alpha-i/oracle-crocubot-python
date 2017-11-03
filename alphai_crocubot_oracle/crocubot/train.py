# Trains the network
# Used by oracle.py

import logging
from timeit import default_timer as timer

import tensorflow as tf

import alphai_crocubot_oracle.bayesian_cost as cost
from alphai_crocubot_oracle.crocubot import PRINT_LOSS_INTERVAL, PRINT_SUMMARY_INTERVAL, MAX_GRADIENT
from alphai_crocubot_oracle.crocubot.model import CrocuBotModel, Estimator

FLAGS = tf.app.flags.FLAGS


# TODO encapsulate the parameters in a ParameterObject
# TODO remove FLAGS usage
def train(topology,
          data_provider,
          tensorflow_path,
          tensorboard_options,
          bin_edges=None,
          restore_path=None):
    """
    :param Toplogy topology:
    :param TrainDataProvider data_provider:
    :param TensorflowPath tensorflow_path:
    :param TensorboardOptions tensorboard_options:
    :param bin_edges:

    :return:
    """
    """ Train network on either MNIST or time series data
    FIXME
    :param Topology topology:
    :param str series_name:
    :return: epoch_loss_list
    """

    _log_topology_parameters_size(topology)

    # Start from a clean graph
    tf.reset_default_graph()
    model = CrocuBotModel(topology, FLAGS)
    model.build_layers_variables()

    # Placeholders for the inputs and outputs of neural networks
    x = tf.placeholder(FLAGS.d_type, shape=[None, topology.n_features_per_series, topology.n_series], name="x")
    y = tf.placeholder(FLAGS.d_type, name="y")

    global_step = tf.Variable(0, trainable=False, name='global_step')
    n_batches = data_provider.get_number_of_batches(FLAGS.batch_size)

    cost_operator = _set_cost_operator(model, x, y, n_batches)
    tf.summary.scalar("cost", cost_operator)
    optimize = _set_training_operator(cost_operator, global_step)

    all_summaries = tf.summary.merge_all()

    saver = tf.train.Saver()

    # Launch the graph
    logging.info("Launching Graph.")
    with tf.Session() as sess:

        is_model_ready = False
        number_of_epochs = FLAGS.n_epochs

        if tensorflow_path.can_restore_model():
            try:
                logging.info("Attempting to load model from {}".format(tensorflow_path.model_restore_path))
                saver.restore(sess, tensorflow_path.model_restore_path)
                logging.info("Model restored.")
                number_of_epochs = FLAGS.n_retrain_epochs
                is_model_ready = True
            except Exception as e:
                logging.warning("Restore file not recovered. reason {}. Training from scratch".format(e))

        if not is_model_ready:
            sess.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter(tensorboard_options.get_log_dir())

        epoch_loss_list = []

        for epoch in range(number_of_epochs):

            data_provider.shuffle_data()

            epoch_loss = 0.
            start_time = timer()

            for batch_number in range(n_batches):  # The randomly sampled weights are fixed within single batch

                batch_data = data_provider.get_batch(batch_number, FLAGS.batch_size)
                batch_x = batch_data.train_x
                batch_y = batch_data.train_y

                if batch_number == 0 and epoch == 0:
                    logging.info("Training {} batches of size {} and {}".format(
                        n_batches,
                        batch_x.shape,
                        batch_y.shape
                    ))

                _, batch_loss, summary_results = sess.run([optimize, cost_operator, all_summaries],
                                                          feed_dict={x: batch_x, y: batch_y})
                epoch_loss += batch_loss

                is_time_to_save_summary = epoch * batch_number % PRINT_SUMMARY_INTERVAL
                if is_time_to_save_summary:
                    summary_index = epoch * n_batches + batch_number
                    summary_writer.add_summary(summary_results, summary_index)

            time_epoch = timer() - start_time

            if epoch_loss != epoch_loss:
                raise ValueError("Found nan value for epoch loss.")

            epoch_loss_list.append(epoch_loss)

            _log_epoch_loss_if_needed(epoch, epoch_loss, number_of_epochs, time_epoch)

        out_path = saver.save(sess, tensorflow_path.session_save_path)
        logging.info("Model saved in file:{}".format(out_path))

    return epoch_loss_list


def _log_epoch_loss_if_needed(epoch, epoch_loss, n_epochs, time_epoch):
    """
    Logs the Loss according to PRINT_LOSS_INTERVAL
    :param int epoch:
    :param float epoch_loss:
    :param int n_epochs:
    :param float time_epoch:
    :return:
    """
    if (epoch % PRINT_LOSS_INTERVAL) == 0:
        msg = "Epoch {} of {} ... Loss: {:.2e}. in {:.2f} seconds."
        logging.info(msg.format(epoch + 1, n_epochs, epoch_loss, time_epoch))


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
    log_predictions = estimator.average_multiple_passes(x, FLAGS.n_train_passes)

    if FLAGS.cost_type == 'bayes':
        operator = cost_object.get_bayesian_cost(log_predictions, labels)
    elif FLAGS.cost_type == 'softmax':
        operator = tf.nn.softmax_cross_entropy_with_logits(logits=log_predictions, labels=labels)
    else:
        raise NotImplementedError

    return tf.reduce_mean(operator)


def _log_topology_parameters_size(topology):
    """Check topology is sensible """

    logging.info("Requested topology: {}".format(topology.layers))

    if topology.n_parameters > 1e7:
        logging.warning("Ambitious number of parameters: {}".format(topology.n_parameters))
    else:
        logging.info("Number of parameters: {}".format(topology.n_parameters))


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
