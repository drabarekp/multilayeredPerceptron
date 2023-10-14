import numpy as np
import SpecialFunctions as sf
from numpy import random


class MlpBase:
    def __init__(self, layers_description):
        self.layers = []
        self.biases = []
        self.activation = lambda x: sf.sigmoid(x)

        for i in range(len(layers_description) - 1):
            self.layers.append(np.random.random((layers_description[i + 1], layers_description[i])))
            self.biases.append(np.random.random(size=layers_description[i + 1]))

    def operation(self, _input):
        if len(_input) != len(self.layers[0][0]):
            raise Exception(f'vector was {len(_input)}, network 1st layer is {len(self.layers[0][0])}')

        for i in range(len(self.layers)):
            _input = self.layers[i] @ _input + self.biases[i]
            _input = self.activation(_input)

        return _input

    def backpropagation(self):
        return None

    def __str__(self):
        result = "layers:\n"
        for i in range(len(self.layers)):
            result += self.layers[i].__str__()
            result += "\n\n"
        result += "biases:\n"
        for i in range(len(self.biases)):
            result += self.biases[i].__str__()
            result += "\n"

        return result
