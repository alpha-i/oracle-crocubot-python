# Trains the network
# Used by oracle.py

import logging
from timeit import default_timer as timer

import tensorflow as tf
import numpy as np

import alphai_crocubot_oracle.bayesian_cost as cost
from alphai_crocubot_oracle.crocubot import PRINT_LOSS_INTERVAL, PRINT_SUMMARY_INTERVAL, MAX_GRADIENT
from alphai_crocubot_oracle.crocubot.model import CrocuBotModel, Estimator

PRINT_KERNEL = True
BOOL_TRUE = True
USE_EFFICIENT_PASSES = True


def train(topology,
          data_provider,
          tensorflow_path,
          tensorboard_options,
          tf_flags):
    """
    :param Toplogy topology:
    :param TrainDataProvider data_provider:
    :param TensorflowPath tensorflow_path:
    :param TensorboardOptions tensorboard_options:
    :param tf_flags:
    :return:
    """

    _log_topology_parameters_size(topology)
    do_retraining = tensorflow_path.can_restore_model()

    # Start from a clean graph
    tf.reset_default_graph()
    is_training = tf.placeholder(tf.bool, name='is_training')

    model = CrocuBotModel(topology, tf_flags, is_training)
    model.build_layers_variables()

    # Placeholders for the inputs and outputs of neural networks
    x_shape = (None, topology.n_series, topology.n_timesteps, topology.n_features)
    x = tf.placeholder(tf_flags.d_type, shape=x_shape, name="x")
    y = tf.placeholder(tf_flags.d_type, name="y")

    global_step = tf.Variable(0, trainable=False, name='global_step')
    n_batches = data_provider.number_of_batches

    cost_operator, log_predict, log_likeli = _set_cost_operator(model, x, y, n_batches, tf_flags, global_step)
    tf.summary.scalar("cost", cost_operator)
    optimize = _set_training_operator(cost_operator, global_step, tf_flags, do_retraining, topology)

    all_summaries = tf.summary.merge_all()

    saver = tf.train.Saver()
    epoch_loss_list = []

    # Launch the graph
    logging.info("Launching Graph.")
    with tf.Session() as sess:

        is_model_ready = False
        number_of_epochs = tf_flags.n_epochs

        if do_retraining:
            if tf_flags.n_retrain_epochs < 1:
                return epoch_loss_list  # Don't waste time loading model
            try:
                logging.info("Attempting to load model from {}".format(tensorflow_path.model_restore_path))
                saver.restore(sess, tensorflow_path.model_restore_path)
                logging.info("Model restored.")
                number_of_epochs = tf_flags.n_retrain_epochs
                is_model_ready = True
            except Exception as e:
                logging.warning("Restore file not recovered. reason {}. Training from scratch".format(e))
        else:
            logging.info("Training new network with fixed random seed")
            tf.set_random_seed(tf_flags.random_seed)  # Ensure behaviour is reproducible
            np.random.seed(tf_flags.random_seed)

        if not is_model_ready:
            sess.run(tf.global_variables_initializer())

        summary_writer = tf.summary.FileWriter(tensorboard_options.get_log_dir())

        for epoch in range(number_of_epochs):

            data_provider.shuffle_data()

            epoch_loss = 0.
            epoch_likeli = 0
            start_time = timer()

            for batch_number in range(n_batches):  # The randomly sampled weights are fixed within single batch

                batch_data = data_provider.get_noisy_batch(batch_number, tf_flags.noise_amplitude)
                batch_features = batch_data.features
                batch_labels = batch_data.labels

                if batch_number == 0 and epoch == 0:
                    logging.info("Training {} batches of size {} and {}".format(
                        n_batches,
                        batch_features.shape,
                        batch_labels.shape
                    ))

                _, batch_loss, batch_likeli, summary_results = \
                    sess.run([optimize, cost_operator, log_likeli, all_summaries],
                             feed_dict={x: batch_features, y: batch_labels, is_training: BOOL_TRUE})
                epoch_loss += batch_loss
                epoch_likeli += batch_likeli

                is_time_to_save_summary = epoch * batch_number % PRINT_SUMMARY_INTERVAL
                if is_time_to_save_summary:
                    summary_index = epoch * n_batches + batch_number
                    summary_writer.add_summary(summary_results, summary_index)

            time_epoch = timer() - start_time

            if epoch_loss != epoch_loss:
                raise ValueError("Found nan value for epoch loss.")

            epoch_loss_list.append(epoch_loss)
            _log_epoch_loss_if_needed(epoch, epoch_loss, epoch_likeli, number_of_epochs, time_epoch,
                                      tf_flags.use_convolution)

        sample_log_predictions = sess.run(log_predict,
                                          feed_dict={x: batch_features, y: batch_labels, is_training: BOOL_TRUE})
        log_network_confidence(sample_log_predictions)
        out_path = saver.save(sess, tensorflow_path.session_save_path)
        logging.info("Model saved in file:{}".format(out_path))

    return epoch_loss_list


def _log_epoch_loss_if_needed(epoch, epoch_loss, log_likelihood, n_epochs, time_epoch, use_convolution):
    """
    Logs the Loss according to PRINT_LOSS_INTERVAL
    :param int epoch:
    :param float epoch_loss:
    :param int n_epochs:
    :param float time_epoch:
    :param bool use_convolution
    :return:
    """
    if (epoch % PRINT_LOSS_INTERVAL) == 0:
        msg = "Epoch {} of {} ... Loss: {:.2e}. LogLikeli: {:.2e} in {:.2f} seconds."
        logging.info(msg.format(epoch + 1, n_epochs, epoch_loss, log_likelihood, time_epoch))

        if PRINT_KERNEL and use_convolution:
            gr = tf.get_default_graph()
            conv1_kernel_val = gr.get_tensor_by_name('conv3d0/kernel:0').eval()
            kernel_shape = conv1_kernel_val.shape
            kernel_sample = conv1_kernel_val.flatten()[0:3]
            logging.info("Sample from first layer {} kernel: {}".format(kernel_shape, kernel_sample))


def _set_cost_operator(crocubot_model, x, labels, n_batches, tf_flags, global_step):

    """
    Set the cost operator
    :param crocubot_model:
    :param x:
    :param labels:
    :param n_batches:
    :param tf_flags:
    :param global step: keep track of how far training has progressed
    :return:
    """

    cost_object = cost.BayesianCost(crocubot_model,
                                    tf_flags.double_gaussian_weights_prior,
                                    tf_flags.wide_prior_std,
                                    tf_flags.narrow_prior_std,
                                    tf_flags.spike_slab_weighting,
                                    n_batches
                                    )

    estimator = Estimator(crocubot_model, tf_flags)

    if USE_EFFICIENT_PASSES:
        log_predictions = estimator.efficient_multiple_passes(x)
    else:
        log_predictions = estimator.average_multiple_passes(x, tf_flags.n_train_passes)

    if tf_flags.cost_type == 'bbalpha':
        cost_operator = cost_object.get_hellinger_cost(x, labels, tf_flags.n_train_passes, estimator)
        log_likelihood = tf.reduce_mean(cost_operator)
    elif tf_flags.cost_type == 'bayes':
        cost_operator = cost_object.get_bayesian_cost(log_predictions, labels, global_step)
        likelihood_op = cost_object.calculate_likelihood(labels, log_predictions)
        log_likelihood = tf.reduce_mean(likelihood_op)
    elif tf_flags.cost_type == 'softmax':
        cost_operator = tf.nn.softmax_cross_entropy_with_logits(logits=log_predictions, labels=labels)
        log_likelihood = tf.reduce_mean(cost_operator)
    else:
        raise NotImplementedError('Unsupported cost type:', tf_flags.cost_type)

    total_cost = tf.reduce_mean(cost_operator)

    return total_cost, log_predictions, log_likelihood


def _log_topology_parameters_size(topology):
    """Check topology is sensible """

    logging.info("Requested topology: {}".format(topology.layers))

    if topology.n_parameters > 1e7:
        logging.warning("Ambitious number of parameters: {}".format(topology.n_parameters))
    else:
        logging.info("Number of parameters: {}".format(topology.n_parameters))


# TODO Create a Provider for training_operator
def _set_training_operator(cost_operator, global_step, tf_flags, do_retraining, topology):
    """ Define the algorithm for updating the trainable variables. """

    if tf_flags.optimisation_method == 'Adam':
        optimizer = tf.train.AdamOptimizer(tf_flags.learning_rate)
        gradients, variables = zip(*optimizer.compute_gradients(cost_operator))
        gradients, _ = tf.clip_by_global_norm(gradients, MAX_GRADIENT)
        optimize = optimizer.apply_gradients(zip(gradients, variables), global_step=global_step)
    elif tf_flags.optimisation_method == 'GDO':
        if tf_flags.partial_retrain and do_retraining:
            final_layer_scope = str(topology.n_layers - 1)
            trainable_var_list = tf.trainable_variables(scope=final_layer_scope)  # mu and sigma of weights and biases
            logging.info("Retraining variables from final layer: {}".format(trainable_var_list[0]))
        else:
            trainable_var_list = None  # By default will train all available variables

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)   # For batch normalisation
        with tf.control_dependencies(update_ops):
            optimizer = tf.train.GradientDescentOptimizer(tf_flags.learning_rate)
            grads_and_vars = optimizer.compute_gradients(cost_operator, var_list=trainable_var_list)
            clipped_grads_and_vars = [(tf.clip_by_value(g, -MAX_GRADIENT, MAX_GRADIENT), v) for g, v in grads_and_vars]
            optimize = optimizer.apply_gradients(clipped_grads_and_vars)

    else:
        raise NotImplementedError("Unknown optimisation method: ", tf_flags.optimisation_method)

    return optimize


def log_network_confidence(log_predictions):
    """  From a sample of predictions, returns the typical confidence applied to a forecast.

    :param nparray log_predictions: multidimensional array of log probabilities [samples, n_forecasts, n_bins]
    :return: null
    """

    predictions = np.exp(log_predictions)
    confidence_values = np.max(predictions, axis=-1).flatten()
    typical_confidence = np.median(confidence_values)

    logging.info('Typical network confidence for a single pass: {}'.format(typical_confidence))
