import numpy as np
from numpy import random


class MlpBase:
    def __init__(self, layersDescription):
        self.layers = []
        for layerSize in layersDescription:
            self.layers.append([random.uniform(0, 1) for _ in range(layerSize)])

    def __str__(self):
        return f"{self.layers}"
