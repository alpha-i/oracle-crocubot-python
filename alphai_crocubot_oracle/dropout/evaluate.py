import logging

import tensorflow as tf
from timeit import default_timer as timer

from alphai_crocubot_oracle.dropout.model import dropout_net

PRINT_SAMPLE_WEIGHTS = True
SAMPLE_WEIGHTS = 'DropoutNetTrain/layer1-conv/weights:0'


def eval_neural_net(predict_x, tf_flags, load_file):
    """ Make the damn prediction.

    :param predict_x:
    :param tf_flags:
    :param load_file:
    :return:
    """

    tf.reset_default_graph()

    x = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='Inputs')
    is_training = tf.placeholder(tf.bool, name='IsTraining')
    logits = dropout_net(x, is_training, scope='DropoutNetTrain', n_classification_bins=tf_flags.n_classification_bins)
    eval_operator = tf.nn.softmax(logits, dim=-1)

    saver = tf.train.Saver()

    with tf.Session() as sess:

        logging.info("Attempting to recover trained network: {}".format(load_file))
        start_time = timer()
        saver.restore(sess, load_file)
        end_time = timer()
        delta_time = end_time - start_time
        logging.info("Loading the model from disk took:{}".format(delta_time))

        prediction = sess.run(eval_operator, feed_dict={x: predict_x, is_training: False})

        if PRINT_SAMPLE_WEIGHTS:
            sample_weights = sess.run(SAMPLE_WEIGHTS).flatten()
            logging.info("Sample of weights from first layer:{}".format(sample_weights[0:9]))
            logging.info("First prediction:{}".format(prediction[0, :]))

    return prediction
