import numpy as np
import functions as sf


# DONE: added seed


class MlpBase:
    def __init__(self, layers_description, _seed):
        np.random.seed(_seed)
        self.layers = []
        self.biases = []
        self.node_values_pre_activation = []

        self.activation = lambda x: sf.sigmoid(x)
        self.activation_derivative = lambda x: sf.sigmoid_derivative(x)
        self.last_layer_activation = lambda x: sf.sigmoid(x)
        self.last_layer_activation_derivative = lambda x: sf.sigmoid_derivative(x)
        self.loss = lambda a, b: sf.mean_squared_error(a, b)
        self.loss_gradient = lambda a, b: sf.mean_squared_error_derivative_a(a, b)

        self.descent_length = 0.1

        for i in range(len(layers_description) - 1):
            self.layers.append(
                np.random.uniform(low=-0.5, high=0.5, size=(layers_description[i + 1], layers_description[i])))

            # self.biases.append(np.random.random(size=layers_description[i + 1]))
            self.biases.append(np.zeros(layers_description[i + 1]))

    def operation(self, _input):
        if len(_input) != len(self.layers[0][0]):
            raise Exception(f'vector was {len(_input)}, network 1st layer is {len(self.layers[0][0])}')

        self.node_values_pre_activation.append(_input)
        for i in range(len(self.layers) - 1):

            _input = self.layers[i] @ _input + self.biases[i]
            self.node_values_pre_activation.append(_input)
            _input = self.activation(_input)

        if len(self.layers) - 1 >= 0:
            _input = self.layers[len(self.layers) - 1] @ _input + self.biases[len(self.layers) - 1]
            self.node_values_pre_activation.append(_input)
            _input = self.last_layer_activation(_input)

        return _input

    def backpropagation(self, output, expected_output):
        error_delta = []

        last_delta = (self.loss_gradient(output, expected_output) *
                      self.last_layer_activation_derivative(self.node_values_pre_activation[self.layer_count()]))
        error_delta.insert(0, last_delta)

        for i in reversed(range(self.layer_count() - 1)):
            current_delta = ((self.layers[i] @ error_delta[0]) *
                             self.activation_derivative(self.node_values_pre_activation[i+1]))

            error_delta.insert(0, current_delta)

        return error_delta

    def descent(self, error_deltas):
        for i in range(self.layer_count()):
            a = error_deltas[i][np.newaxis].transpose()
            b = self.activation(self.node_values_pre_activation[i])[np.newaxis]
            gradient = a @ b
            self.layers[i] -= self.descent_length * gradient

    def learn(self, _input, expected_output):
        output = self.operation(_input)
        deltas = self.backpropagation(output, expected_output)
        self.descent(deltas)

        return deltas

    def layer_count(self):
        return len(self.layers)

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
