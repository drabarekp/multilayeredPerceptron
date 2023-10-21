from read_file import *
from MlpBase import MlpBase
import numpy as np

np.seterr(all='raise')

# temporary function for testing and getting feedback from mlp
def peek():
    mlp = MlpBase([2, 1], _seed=1002)
    """
    output = mlp.operation([0.5, 0.5, 0.5])
    print(f"out={output}")
    """

    # test()
    for i in range(10000):
        expected_output = [0.2]
        deltas = mlp.learn(np.array([0.3, 0.6]), np.array(expected_output))
        #try:
            #deltas = mlp.learn(np.array([0.3, 0.3]), np.array(expected_output))
        #except Exception:
            #break

        if i % 1000 == 0:
            output = mlp.operation(np.array([0.3, 0.6]))
            print(f"{mlp.loss(output, np.array(expected_output))}, {output}")



# def test():
#     val = np.array([1, 2, 3]) * np.transpose(np.array([5, 4, 3]))
#     print(val)


# main
if __name__ == '__main__':
    # read_classification('data_classification/data.simple.test.100.csv')
    # read_regression('data_regression/data.activation.test.100.csv')
    peek()
