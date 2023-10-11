import numpy as np
from numpy import random


class MlpBase:
    def __init__(self, layersDescription):
        self.layers = []
        for i in range(len(layersDescription) - 1):
            self.layers.append(np.random.random((layersDescription[i], layersDescription[i + 1])))

    def __str__(self):
        result = ""
        for i in range(len(self.layers)):
            result += self.layers[i].__str__()
            result += "\n\n"
        return result
