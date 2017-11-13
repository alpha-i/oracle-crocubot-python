import datetime

import numpy as np
import tensorflow as tf

import alphai_crocubot_oracle.crocubot.evaluate as crocubot_eval
import alphai_crocubot_oracle.crocubot.train as crocubot_train

from alphai_crocubot_oracle.crocubot.helpers import TensorflowPath, TensorboardOptions
from alphai_crocubot_oracle.crocubot.model import CrocuBotModel
from alphai_crocubot_oracle.data.providers import TrainDataProviderForDataSource
from alphai_crocubot_oracle.helpers import printtime, execute_and_get_duration

import examples.iotools as io
from examples.benchmark.helpers import print_time_info, print_accuracy, _calculate_accuracy
from examples.helpers import D_TYPE, load_default_topology


def run_timed_benchmark_mnist(series_name, tf_flags, do_training):

    topology = load_default_topology(series_name, tf_flags)

    execution_time = datetime.datetime.now()
    save_file = io.build_check_point_filename(series_name, topology, tf_flags)

    @printtime(message="Training MNIST with _do_training: {}".format(int(do_training)))
    def _do_training():
        if do_training:
            data_provider = TrainDataProviderForDataSource(series_name,
                                                           D_TYPE,
                                                           tf_flags.n_training_samples_benchmark,
                                                           tf_flags.batch_size,
                                                           True
                                                           )

            tensorflow_path = TensorflowPath(save_file)
            tensorboard_options = TensorboardOptions(tf_flags.tensorboard_log_path,
                                                     tf_flags.learning_rate,
                                                     tf_flags.batch_size,
                                                     execution_time
                                                     )

            crocubot_train.train(topology,
                                 data_provider,
                                 tensorflow_path,
                                 tensorboard_options,
                                 tf_flags
                                 )
        else:
            tf.reset_default_graph()
            model = CrocuBotModel(topology, tf_flags)
            model.build_layers_variables()

    train_time, _ = execute_and_get_duration(_do_training)
    print("Training complete.")

    eval_time, metrics = execute_and_get_duration(evaluate_network, topology, series_name,
                                                  tf_flags.batch_size, save_file, tf_flags)

    accuracy = _calculate_accuracy(metrics["results"])
    print('Metrics:')
    print_accuracy(metrics, accuracy)

    print_time_info(train_time, eval_time)


@printtime(message="Evaluation of Mnist Serie")
def evaluate_network(topology, series_name, batch_size, save_file, tf_flags):

    data_provider = TrainDataProviderForDataSource(series_name, D_TYPE, tf_flags.n_prediction_sample, batch_size, False)

    test_features, test_labels = data_provider.get_batch(1)

    binned_outputs = crocubot_eval.eval_neural_net(test_features, topology, tf_flags, save_file)

    return evaluate_mnist(binned_outputs, binned_outputs.shape[1], test_labels)


def evaluate_mnist(binned_outputs, n_samples, test_labels):
    binned_outputs = np.mean(binned_outputs, axis=0)  # Average over passes
    predicted_indices = np.argmax(binned_outputs, axis=-1).flatten()
    true_indices = np.argmax(test_labels, axis=-1).flatten()

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

