import numpy as np

TIME_LIMIT = 600


def print_time_info(train_time, eval_time):

    print('Training took', str.format('{0:.2f}', train_time), "seconds")
    print('Evaluation took', str.format('{0:.2f}', eval_time), "seconds")

    if train_time > TIME_LIMIT:
        print('** Training took ', str.format('{0:.2f}', train_time - TIME_LIMIT),
              ' seconds too long - DISQUALIFIED! **')


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
