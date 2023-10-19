import numpy as np


def read_classification(path):
    my_data = np.genfromtxt(path, delimiter=',', skip_header=1,
                            dtype=[('x', np.float64), ('y', np.float64), ('cls', np.int32)])
    return my_data
    # print(my_data)
    # print(my_data[6])
    # print(my_data[6][0])


def read_regression(path):
    my_data = np.genfromtxt(path, delimiter=',', skip_header=1,
                            dtype=[('x', np.float64), ('y', np.float64)])
    return my_data
    # print(my_data)
    # print(my_data[6])
    # print(my_data[6][0])
