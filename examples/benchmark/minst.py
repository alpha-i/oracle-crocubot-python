import datetime

import numpy as np
import tensorflow as tf

from alphai_crocubot_oracle import iotools as io
from alphai_crocubot_oracle.crocubot import train as crocubot_train
from alphai_crocubot_oracle.crocubot.helpers import TensorflowPath, TensorboardOptions
from alphai_crocubot_oracle.crocubot.model import CrocuBotModel
from alphai_crocubot_oracle.crocubot import evaluate as eval
from alphai_crocubot_oracle.data.providers import TrainDataProviderForDataSource
from alphai_crocubot_oracle.helpers import printtime
from examples.helpers import FLAGS, D_TYPE, load_default_topology


def run_timed_benchmark_mnist(series_name, do_training):

    topology = load_default_topology(series_name)

    batch_size = 200
    execution_time = datetime.datetime.now()

    @printtime(message="Training with do_train: {}".format(int(do_training)))
    def _do_training():
        if do_training:
            data_provider = TrainDataProviderForDataSource(series_name, D_TYPE, batch_size, True)
            save_path = io.build_check_point_filename(series_name, topology)
            tensorflow_path = TensorflowPath(save_path)
            tensorboard_options = TensorboardOptions(FLAGS.tensorboard_log_path,
                                                     FLAGS.learning_rate,
                                                     FLAGS.batch_size,
                                                     execution_time
                                                     )

            crocubot_train.train(topology,
                                 data_provider,
                                 tensorflow_path,
                                 tensorboard_options
                                 )
        else:
            tf.reset_default_graph()
            model = CrocuBotModel(topology)
            model.build_layers_variables()

    _do_training()

    print("Training complete.")
    metrics = evaluate_network(topology, series_name)

    accuracy = _calculate_accuracy(metrics["results"])

    print('Metrics:')
    print_accuracy(metrics, accuracy)

    return accuracy, metrics


@printtime(message="Evaluation of Mnist Serie")
def evaluate_network(topology, series_name):  # bin_dist not used in MNIST case
    data_provider = TrainDataProviderForDataSource(series_name, D_TYPE, FLAGS.batch_size * 2, False)

    test_features, test_labels = data_provider.get_batch(1, FLAGS.batch_size)

    save_file = io.build_check_point_filename(series_name, topology)

    binned_outputs = eval.eval_neural_net(test_features, topology, save_file)
    n_samples = binned_outputs.shape[1]

    return evaluate_mnist(binned_outputs, n_samples, test_labels)


def evaluate_mnist(binned_outputs, n_samples, test_labels):
    binned_outputs = np.mean(binned_outputs, axis=0)  # Average over passes
    predicted_indices = np.argmax(binned_outputs, axis=2)
    true_indices = np.argmax(test_labels, axis=2)
    print("Example forecasts:", binned_outputs[0:5, 0, :])
    print("Example outcomes", test_labels[0:5, 0, :])
    print("Total test samples:", n_samples)
    results = np.equal(predicted_indices, true_indices)
    forecasts = np.zeros(n_samples)
    p_success = []
    p_fail = []
    for i in range(n_samples):
        true_index = true_indices[i]
        forecasts[i] = binned_outputs[i, 0, true_index]

        if true_index == predicted_indices[i]:
            p_success.append(forecasts[i])
        else:
            p_fail.append(forecasts[i])
    log_likelihood_per_sample = np.mean(np.log(forecasts))
    median_probability = np.median(forecasts)

    metrics = {}
    metrics["results"] = results
    metrics["log_likelihood_per_sample"] = log_likelihood_per_sample
    metrics["median_probability"] = median_probability
    metrics["mean_p_success"] = np.mean(np.stack(p_success))
    metrics["mean_p_fail"] = np.mean(np.stack(p_fail))
    metrics["mean_p"] = np.mean(np.stack(forecasts))
    metrics["min_p_fail"] = np.min(np.stack(p_fail))

    return metrics


def print_accuracy(metrics, accuracy):

    theoretical_max_log_likelihood_per_sample = np.log(0.5)*(1 - accuracy)

    print('MNIST accuracy of ', accuracy * 100, '%')
    print('Log Likelihood per sample of ', metrics["log_likelihood_per_sample"])
    print('Theoretical limit for given accuracy ', theoretical_max_log_likelihood_per_sample)
    print('Median probability assigned to true outcome:', metrics["median_probability"])
    print('Mean probability assigned to forecasts:', metrics["mean_p"])
    print('Mean probability assigned to successful forecast:', metrics["mean_p_success"])
    print('Mean probability assigned to unsuccessful forecast:', metrics["mean_p_fail"])
    print('Min probability assigned to unsuccessful forecast:', metrics["min_p_fail"])

    return accuracy


def _calculate_accuracy(results):
    total_tests = len(results)
    correct = np.sum(results)
    return correct / total_tests

