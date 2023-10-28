import numpy as np

from DataNormalizator import DataNormalizator


def read_classification(path):
    data = np.genfromtxt(path, delimiter=',', skip_header=1, dtype=np.float64)
    hot_vals = round(max(data[:, 2]))
    data_input = data[:, :2]
    raw_output = data[:, 2:]
    data_output = np.zeros((data.shape[0], hot_vals))
    for i in range(data.shape[0]):
        data_output[i][round(raw_output[i][0]) - 1] = 1  # one-hot encoding
    return data_input, data_output


def read_regression(path):
    dn = DataNormalizator()
    data = np.genfromtxt(path, delimiter=',', skip_header=1, dtype=np.float64)

    data_input = np.reshape(data[:, 0], (-1, 1))
    data_output = np.reshape(data[:, 1], (-1, 1))

    return data_input, data_output


# those have to be in the same function
def normalize_regression(train_in, train_out, test_in, test_out):
    dn = DataNormalizator()
    start_input = min(train_in)
    end_input = max(train_in)
    start_output = min(train_out)
    end_output = max(train_out)

    return (dn.linear_normalize_into_unit_with_range(train_in, start_input, end_input),
            dn.linear_normalize_into_unit_with_range(train_out, start_output, end_output),
            dn.linear_normalize_into_unit_with_range(test_in, start_input, end_input),
            dn.linear_normalize_into_unit_with_range(test_out, start_output, end_output))
