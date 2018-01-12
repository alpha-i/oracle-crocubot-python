import tensorflow as tf
import tensorflow.contrib.slim as slim


def dropout_net(inputs, is_training, scope='dropout_net', n_classification_bins=10):
    with tf.variable_scope(scope, 'dropout_net'):

        # First Group: Convolution + Pooling 28x28x1 => 28x28x20 => 14x14x20
        net = slim.conv2d(inputs, 20, [5, 5], padding='SAME', scope='layer1-conv')
        net = slim.max_pool2d(net, 2, stride=2, scope='layer2-max-pool')

        # Second Group: Convolution + Pooling 14x14x20 => 10x10x40 => 5x5x40
        net = slim.conv2d(net, 40, [5, 5], padding='VALID', scope='layer3-conv')
        net = slim.max_pool2d(net, 2, stride=2, scope='layer4-max-pool')

        # Reshape: 5x5x40 => 1000x1
        net = tf.reshape(net, [-1, 5*5*40])

        # Fully Connected Layer: 1000x1 => 1000x1
        net = slim.fully_connected(net, 1000, scope='layer5')
        net = slim.dropout(net, is_training=is_training, scope='layer5-dropout')

        # Second Fully Connected: 1000x1 => 1000x1
        net = slim.fully_connected(net, 1000, scope='layer6')
        net = slim.dropout(net, is_training=is_training, scope='layer6-dropout')

        # Output Layer: 1000x1 => nx1
        net = slim.fully_connected(net, n_classification_bins, scope='output')
        net = slim.dropout(net, is_training=is_training, scope='output-dropout')

        return net


def no_dropout_net(inputs, is_training, scope='dropout_net', n_classification_bins=10):
    with tf.variable_scope(scope, 'dropout_net'):

        # First Group: Convolution + Pooling 28x28x1 => 28x28x40 => 14x14x40
        net = slim.conv2d(inputs, 40, [5, 5], padding='SAME', scope='layer1-conv')
        net = slim.max_pool2d(net, 2, stride=2, scope='layer2-max-pool')

        # Second Group: Convolution + Pooling 14x14x40 => 14x14x40 => 13x13x40
        net = slim.conv2d(net, 40, [5, 5], padding='SAME', scope='layer3-conv')
        net = slim.max_pool2d(net, 2, stride=1, scope='layer4-max-pool')

        # Third Group: Convolution only. 13x13x30 => 14x14x20
        net = slim.conv2d(net, 40, [5, 5], padding='SAME', scope='layer5-conv')

        # Fourth Group: Convolution + Pooling 13x13x40 => 13x13x40 => 6*6*40
        net = slim.conv2d(net, 40, [5, 5], padding='SAME', scope='layer7-conv')
        net = slim.max_pool2d(net, 2, stride=2, scope='layer8-max-pool')

        # Reshape: 7x7x40 =>
        n_fully_connected = 6*6*40
        net = tf.reshape(net, [-1, n_fully_connected])

        # Fully Connected Layer: 1000x1 => 1000x1
        net = slim.fully_connected(net, n_fully_connected, scope='layer9')
        #  net = slim.dropout(net, is_training=is_training, scope='layer9-dropout')

        # Second Fully Connected: 1000x1 => 1000x1
        net = slim.fully_connected(net, n_fully_connected, scope='layer10')
        #  net = slim.dropout(net, is_training=is_training, scope='layer10-dropout')

        # Output Layer: 1000x1 => nx1
        net = slim.fully_connected(net, n_classification_bins, scope='output')
        #  net = slim.dropout(net, is_training=is_training, scope='output-dropout')

        return net

#  144000 values, but the requested shape requires a multiple of 1960
# 144000 / 400 = 360 units per sample, expected 1960
# 288000 / 400 = 720 units per sample, expected 980
# 576000 / 400 = 1440 units per sample, but the requested shape requires a multiple of 1960
