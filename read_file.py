import numpy as np


def read_classification(path):
    data = np.genfromtxt(path, delimiter=',', skip_header=1, dtype=np.float64)
    hot_vals = round(max(data[:, 2]))
    data_input = data[:, :2]
    raw_output = data[:, 2:]
    data_output = np.zeros((data.shape[0], hot_vals))
    for i in range(data.shape[0]):
        data_output[i][round(raw_output[i][0]) - 1] = 1  # one-hot encoding
    return data_input, data_output

    # print(my_data)
    # print(my_data[6])
    # print(my_data[6][0])


def read_regression(path):
    data = np.genfromtxt(path, delimiter=',', skip_header=1, dtype=np.float64)
    data_input = np.reshape(data[:, 0], (-1, 1))
    data_output = np.reshape(data[:, 1], (-1, 1))
    return data_input, data_output

    # print(my_data)
    # print(my_data[6])
    # print(my_data[6][0])
