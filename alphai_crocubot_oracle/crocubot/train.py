# Trains the network
# Used by oracle.py


import logging
from timeit import default_timer as timer

import tensorflow as tf

import alphai_crocubot_oracle.bayesian_cost as cost
from alphai_crocubot_oracle.crocubot.model import CrocuBotModel, Estimator
import alphai_crocubot_oracle.iotools as io

FLAGS = tf.app.flags.FLAGS
PRINT_LOSS_INTERVAL = 1
PRINT_SUMMARY_INTERVAL = 1

def make_hparam_string(learning_rate, batch_size):
    return "lr={}_bs={}".format(learning_rate, batch_size)


def train(topology, data_source, train_x=None, train_y=None, bin_edges=None, save_path=None, restore_path=None):
    """ Train network on either MNIST or time series data

    :param Topology topology:
    :param str data_source:
    :return: epoch_loss_list
    """

    split_save_path = save_path.split("/")
    run_id = split_save_path[-1]
    hparam_string = make_hparam_string(FLAGS.learning_rate, FLAGS.batch_size)

    log_dir = "{}/{}/{}".format(FLAGS.log_path, hparam_string, run_id)

    logging.info("Starting run for hyperparameters: {}".format(hparam_string))

    if topology.n_parameters > 1e7:
        logging.warning("Ambitious number of parameters: {}".format(topology.n_parameters))
    else:
        logging.info("Number of parameters: {}".format(topology.n_parameters))

    # Start from a clean graph
    tf.reset_default_graph()
    model = CrocuBotModel(topology, FLAGS)
    model.build_layers_variables()

    use_data_loader = True if train_x is None else False

    # Placeholders for the inputs and outputs of neural networks
    x = tf.placeholder(FLAGS.d_type, shape=[None, topology.n_features_per_series, topology.n_series], name="x")
    y = tf.placeholder(FLAGS.d_type, name="y")

    global_step = tf.Variable(0, trainable=False, name='global_step')

    n_batches = int(FLAGS.n_training_samples / FLAGS.batch_size)

    cost_operator = _set_cost_operator(model, x, y, n_batches)
    tf.summary.scalar("cost", cost_operator)

    training_operator = tf.train.AdamOptimizer(FLAGS.learning_rate).minimize(cost_operator, global_step=global_step)

    all_summaries = tf.summary.merge_all()

    model_initialiser = tf.global_variables_initializer()

    if save_path is None:
        save_path = io.load_file_name(data_source, topology)
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

        summary_writer = tf.summary.FileWriter(log_dir)

        epoch_loss_list = []
        for epoch in range(n_epochs):

            epoch_loss = 0.
            start_time = timer()
            logging.info("Training epoch {} of {}".format(epoch, n_epochs))

            for b in range(n_batches):  # The randomly sampled weights are fixed within single batch

                if use_data_loader:
                    batch_x, batch_y = io.load_training_batch(data_source, batch_number=b, batch_size=FLAGS.batch_size, bin_edges=bin_edges)
                else:
                    lo_index = b * FLAGS.batch_size
                    hi_index = lo_index + FLAGS.batch_size
                    batch_x = train_x[lo_index:hi_index, :]
                    batch_y = train_y[lo_index:hi_index, :]

                if b == 0 and epoch == 0:
                    logging.info("Training {} batches of size {} and {}".format(n_batches, batch_x.shape, batch_y.shape))

                _, batch_loss, summ = sess.run([training_operator, cost_operator, all_summaries], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += batch_loss

                index = epoch * n_batches + b
                summary_writer.add_summary(summ, index)

            time_epoch = timer() - start_time

            epoch_loss_list.append(epoch_loss)

            if (epoch % PRINT_LOSS_INTERVAL) == 0:
                msg = 'Epoch ' + str(epoch) + " loss:" + str.format('{0:.2e}', epoch_loss) + " in " + str.format('{0:.2f}', time_epoch) + " seconds"
                logging.info(msg)


        out_path = saver.save(sess, save_path)
        logging.info("Model saved in file:{}".format(out_path))

    return epoch_loss_list


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
    predictions, _ = estimator.average_multiple_passes(x, FLAGS.n_train_passes)

    if FLAGS.cost_type == 'bayes':
        operator = cost_object.get_bayesian_cost(predictions, labels)
    elif FLAGS.cost_type == 'hellinger':
        operator = cost.get_hellinger_cost(predictions, labels)
    elif FLAGS.cost_type == 'softmax':
        operator = tf.nn.softmax_cross_entropy_with_logits(logits=predictions, labels=labels)
    else:
        raise NotImplementedError

    return tf.reduce_mean(operator)
