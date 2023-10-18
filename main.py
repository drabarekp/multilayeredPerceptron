# main function of mlp app
from MlpBase import MlpBase
import numpy as np

# temporary function for testing and getting feedback from mlp
def peek():
    mlp = MlpBase([10, 10, 10, 5, 1])
    """
    output = mlp.operation([0.5, 0.5, 0.5])
    print(f"out={output}")
    """

    #test()

    for i in range(10000):
        mlp.learn(np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]), np.array([0.5]))
        if i % 1000 == 0:
            output = mlp.operation(np.array([0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3, 0.3]))
            print(f"{mlp.loss(output, np.array([0.5]))}, {output}")


def test():
    val = np.array([1, 2, 3]) * np.transpose(np.array([5, 4, 3]))
    print(val)


# main
if __name__ == '__main__':
    peek()
