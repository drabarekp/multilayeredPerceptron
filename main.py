# main function of mlp app
from MlpBase import MlpBase


# temporary function for testing and getting feedback from mlp
def peek():
    mlp = MlpBase([3, 4, 1])
    output = mlp.operation([0.5, 0.5, 0.5])
    print(f"out={output}")


# main
if __name__ == '__main__':
    peek()
