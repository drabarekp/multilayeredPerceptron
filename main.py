from read_file import *
from MlpBase import MlpBase
import numpy as np
import functions as fn
from page import plot_iterations

np.seterr(all='raise')


# temporary function for testing and getting feedback from mlp
def peek(data_input, data_output):
    mlp = MlpBase([2, 8, 8, 3], _seed=1002)
    """
    output = mlp.operation([0.5, 0.5, 0.5])
    print(f"end={output}")
    """

    # test()
    data_size = data_input.shape[0]
    for i in range(100000):
        # expected_output = []
        # deltas = mlp.learn(np.array([0.3, 0.6]), np.array(expected_output))
        # try:
        #   deltas = mlp.learn(np.array([0.3, 0.3]), np.array(expected_output))
        # except Exception:
        #   break
        # if i % 1000 == 0:
        #     output = mlp.operation(np.array([0.3, 0.6]))
        #     print(f"{mlp.loss(output, np.array(expected_output))}, {output}")

        pos = i % data_size
        deltas = mlp.learn(data_input[pos], data_output[pos])

        if i % 1000 == 0:
            output = mlp.operation(data_input[pos])
            print(f"{mlp.loss(output, data_output[pos])}, {output}")


# def test_np():
#     val = np.array([1, 2, 3]) * np.transpose(np.array([5, 4, 3]))
#     print(val)


# def test_fn():
#     a = np.array([1, 2, 3])
#     b = np.array([4, 5, 7])
#
#     print(fn.sigmoid(a[1]))
#     print(fn.relu(a[1]))
#     print(fn.arctan(a[1]))
#
#     print(fn.sigmoid_derivative(a[1]))
#     print(fn.relu_derivative(a[1]))
#     print(fn.arctan_derivative(a[1]))
#
#     print(fn.sigmoid(a))
#     print(fn.relu(a))
#     print(fn.arctan(a))
#     print(fn.mean_squared_error(a, b))
#     print(fn.mean_absolute_error(a, b))
#
#     print(fn.sigmoid_derivative(a))
#     print(fn.relu_derivative(a))
#     print(fn.arctan_derivative(a))
#     print(fn.mean_squared_error_derivative(a, b))
#     print(fn.mean_absolute_error_derivative(a, b))


# main
if __name__ == '__main__':
    d_in, d_out = read_classification('data_classification/data.three_gauss.train.100.csv')
    # d_in, d_out = read_regression('data_regression/data.activation.test.100.csv')

    # data = [
    #     [[0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4],
    #      [0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3],
    #      [0.1, 0.2, -0.1, 0.2, -0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, -0.2]],
    #     [[0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4, 0.1, 0.2, 0.3, 0.4],
    #      [0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3, 0.1, 0.2, 0.3],
    #      [0.1, 0.2, -0.1, 0.2, -0.1, 0.2, 0.1, 0.2, 0.1, 0.2, 0.1, -0.2]]
    # ]
    # layers = [1, 4, 2, 1]
    # plot_iterations(data, layers)

    # a = np.array([1, 2, 3])
    # b = np.array([4, 5, 7])
    # print(fn.relu(a))
    peek(d_in, d_out)
