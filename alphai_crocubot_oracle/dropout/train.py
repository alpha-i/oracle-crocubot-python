import logging

import tensorflow as tf
import numpy as np

from alphai_crocubot_oracle.dropout.model import dropout_net
PRINT_INTERVAL = 5000
MOMENTUM = 0.5


def train(data_provider, tensorflow_path, tf_flags):
    """

    :param data_provider:
    :param tensorflow_path:
    :param tf_flags:
    :return:
    """

    # Create the placeholder tensors for the input images (x), the training labels (y_actual)
    # and whether or not dropout is active (is_training)
    tf.reset_default_graph()
    n_bins = tf_flags.n_classification_bins

    x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='Inputs')
    y_actual = tf.placeholder(tf.float32, shape=[None, n_bins], name='Labels')
    is_training = tf.placeholder(tf.bool, name='IsTraining')

    # Pass the inputs into dropout_net, outputting the logits
    logits = dropout_net(x, is_training, scope='DropoutNetTrain', n_classification_bins=n_bins)
    eval_operator = tf.nn.softmax(logits, dim=-1)

    cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_actual))
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(y_actual, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    train_step = tf.train.MomentumOptimizer(tf_flags.learning_rate, MOMENTUM).minimize(cross_entropy)

    loss_summary = tf.summary.scalar('loss', cross_entropy)
    accuracy_summary = tf.summary.scalar('accuracy', accuracy)
    number_of_epochs = tf_flags.n_epochs

    saver = tf.train.Saver()

    x_all = data_provider._train_data.features
    y_all = data_provider._train_data.labels

    # Launch the graph
    logging.info("Launching Graph.")
    with tf.Session() as sess:

        is_model_ready = False
        if tensorflow_path.can_restore_model():
            try:
                logging.info("Attempting to load model from {}".format(tensorflow_path.model_restore_path))
                saver.restore(sess, tensorflow_path.model_restore_path)
                logging.info("Model restored.")
                number_of_epochs = tf_flags.n_retrain_epochs
                is_model_ready = True
            except Exception as e:
                logging.warning("Restore file not recovered. reason {}. Training from scratch".format(e))

        if not is_model_ready:
            sess.run(tf.global_variables_initializer())

        step = 0
        for epoch in range(number_of_epochs):

            data_provider.shuffle_data()

            for batch_number in range(data_provider.number_of_batches):

                batch_data = data_provider.get_batch(batch_number)
                batch_features = batch_data.features
                batch_labels = batch_data.labels

                summary, _ = sess.run([loss_summary, train_step],
                                      feed_dict={x: batch_features, y_actual: batch_labels, is_training: True})

                if step % PRINT_INTERVAL == 0:
                    summary, acc = sess.run([accuracy_summary, accuracy],
                                            feed_dict={x: x_all, y_actual: y_all, is_training: False})
                    logging.info("Steps completed: {}. Training Accuracy: {}".format(step, acc * 100))
                step += 1

        test_feature = np.expand_dims(0.98 * batch_features[0, :], axis=0)
        test_prediction = sess.run(eval_operator, feed_dict={x: test_feature, is_training: False})
        logging.info("Example prediction:{}".format(test_prediction))

        out_path = saver.save(sess, tensorflow_path.session_save_path)
        logging.info("Model saved in file:{}".format(out_path))


def resize_time_series(features):
    """ Puts time series [samples, series, timesteps, features] into fixed network format

    :param features:
    :return:
    """

    return np.reshape(features, [-1, 28, 28, 1])
