# Trains the network
# Used by oracle.py


import logging
from timeit import default_timer as timer

import tensorflow as tf

import alphai_crocubot_oracle.bayesian_cost as cost
from alphai_crocubot_oracle.crocubot.model import CrocuBotModel, Estimator
import alphai_crocubot_oracle.iotools as io

FLAGS = tf.app.flags.FLAGS
PRINT_LOSS_INTERVAL = 20


def train(topology, data_source, flags, train_x=None, train_y=None, bin_edges=None, save_path=None):
    """ Train network on either MNIST or time series data

    :param Topology topology:
    :param str data_source:
    :return: epoch_loss_list
    """

    # Start from a clean graph
    tf.reset_default_graph()
    model = CrocuBotModel(topology, flags)
    model.build_layers_variables()

    if train_x is None:
        use_data_loader = True
    else:
        use_data_loader = False

    # Placeholders for the inputs and outputs of neural networks
    x = tf.placeholder(FLAGS.d_type, shape=[None, topology.n_features_per_series, topology.n_series])
    y = tf.placeholder(FLAGS.d_type)
    global_step = tf.Variable(0, trainable=False, name='global_step')

    cost_operator = _set_cost_operator(model, x, y)
    training_operator = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cost_operator, global_step=global_step)
    model_initialiser = tf.global_variables_initializer()

    n_batches = int(FLAGS.n_training_samples / FLAGS.batch_size)
    if save_path is None:
        save_path = io.load_file_name(data_source, topology)
    saver = tf.train.Saver()

    # Launch the graph
    logging.info("Launching Graph.")
    with tf.Session() as sess:

        if FLAGS.resume_training:
            try:
                saver.restore(sess, save_path)
                logging.info("Model restored.")
            except:
                logging.warning("Previous save file not found. Training from scratch")
                sess.run(model_initialiser)
        else:
            sess.run(model_initialiser)

        for epoch in range(FLAGS.n_epochs):

            epoch_loss = 0.
            epoch_loss_list = []
            start_time = timer()

            for b in range(n_batches):  # The randomly sampled weights are fixed within single batch
                if use_data_loader:
                    batch_x, batch_y = io.load_training_batch(data_source, batch_number=b, batch_size=FLAGS.batch_size, labels_per_series=topology.n_classification_bins, bin_edges=bin_edges)
                else:
                    lo_index = b * FLAGS.batch_size
                    hi_index = lo_index + FLAGS.batch_size
                    batch_x = train_x[lo_index:hi_index, :]
                    batch_y = train_y[lo_index:hi_index, :]

                _, batch_loss = sess.run([training_operator, cost_operator], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += batch_loss
                model.increment_random_seed()

            time_epoch = timer() - start_time
            epoch_loss_list.append(epoch_loss)

            if (epoch % PRINT_LOSS_INTERVAL) == 0:
                logging.info('Epoch', epoch, "loss:", str.format('{0:.2e}', epoch_loss), "in", str.format('{0:.2f}', time_epoch), "seconds")

        out_path = saver.save(sess, save_path)
        logging.info("Model saved in file:", out_path)

    return epoch_loss_list


def _set_cost_operator(crocubot_model, x, labels):
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
                                    FLAGS.spike_slab_weighting
                                    )

    estimator = Estimator(crocubot_model, FLAGS)
    predictions = estimator.average_multiple_passes(x, FLAGS.n_train_passes)

    if FLAGS.cost_type == 'bayes':
        operator = cost_object.get_bayesian_cost(predictions, labels)
    elif FLAGS.cost_type == 'hellinger':
        operator = cost.get_hellinger_cost(predictions, labels)
    elif FLAGS.cost_type == 'softmax':
        operator = tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=labels)
    else:
        raise NotImplementedError

    return tf.reduce_mean(operator)
