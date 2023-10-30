import copy
import numpy as np
# import functions as sf


class MlpBase:
    def __init__(self,
                 layers_description,
                 _seed,
                 activation,
                 activation_derivative,
                 last_layer_activation,
                 last_layer_activation_derivative,
                 loss,
                 loss_gradient,
                 descent_length):

        np.random.seed(_seed)
        self.layers_description = layers_description
        self.layers = []
        self.biases = []
        self.node_values_pre_activation = []
        self.node_values_activated = []

        self.activation = activation
        self.activation_derivative = activation_derivative
        self.last_layer_activation = last_layer_activation
        self.last_layer_activation_derivative = last_layer_activation_derivative
        self.loss = loss
        self.loss_gradient = loss_gradient

        self.descent_length = descent_length

        for i in range(len(layers_description) - 1):
            self.layers.append(
                np.random.uniform(low=-1, high=1, size=(layers_description[i + 1], layers_description[i])))

            # self.biases.append(np.random.random(size=layers_description[i + 1]))
            self.biases.append(np.random.uniform(low=-0.5, high=0.5, size=(layers_description[i + 1])))

    def operation(self, _input):
        self.node_values_pre_activation = []
        self.node_values_activated = []

        if len(_input) != len(self.layers[0][0]):
            raise Exception(f'vector was {len(_input)}, network 1st layer is {len(self.layers[0][0])}')

        self.node_values_pre_activation.append(_input)
        self.node_values_activated.append(_input)
        for i in range(len(self.layers) - 1):
            _input = self.layers[i] @ _input + self.biases[i]
            self.node_values_pre_activation.append(_input)
            _input = self.activation(_input)
            self.node_values_activated.append(_input)

        if len(self.layers) - 1 >= 0:
            _input = self.layers[len(self.layers) - 1] @ _input + self.biases[len(self.layers) - 1]
            self.node_values_pre_activation.append(_input)
            _input = self.last_layer_activation(_input)
            self.node_values_activated.append(_input)

        return _input

    def backpropagation(self, output, expected_output):
        error_delta = []
        last_delta = (self.loss_gradient(output, expected_output) *
                      self.last_layer_activation_derivative(self.node_values_pre_activation[self.layer_count()]))
        error_delta.insert(0, last_delta)

        for i in reversed(range(self.layer_count() - 1)):
            a = self.layers[i + 1].transpose()
            b = error_delta[0]
            c = self.activation_derivative(self.node_values_pre_activation[i + 1])
            current_delta = (a @ b) * c

            error_delta.insert(0, current_delta)

        return error_delta

    def descent(self, error_deltas):
        for i in range(self.layer_count()):
            a = error_deltas[i][np.newaxis].transpose()
            b = self.node_values_activated[i][np.newaxis]

            gradient = a @ b
            self.layers[i] -= self.descent_length * gradient
            self.biases[i] -= self.descent_length * error_deltas[i]

    def learn(self, _input, expected_output):
        output = self.operation(_input)
        deltas = self.backpropagation(output, expected_output)
        self.descent(deltas)

        return output

    def learn_iteration(self, train_input, train_output, test_input, test_output):
        old_layers = copy.deepcopy(self.layers)
        old_biases = copy.deepcopy(self.biases)

        train_size = train_input.shape[0]
        test_size = test_input.shape[0]
        train_error = 0
        test_error = 0

        for pos in range(train_size):
            output = self.learn(train_input[pos], train_output[pos])
            train_error += self.loss(output, train_output[pos])
        train_error /= train_size

        for pos in range(test_size):
            output = self.operation(test_input[pos])
            test_error += self.loss(output, test_output[pos])
        test_error /= test_size

        current_layers = copy.deepcopy(self.layers)
        current_biases = copy.deepcopy(self.biases)
        delta_layers = np.subtract(current_layers, old_layers).tolist()
        delta_biases = np.subtract(current_biases, old_biases).tolist()

        current_biases.insert(0, np.zeros(self.layers_description[0]))
        delta_biases.insert(0, np.zeros(self.layers_description[0]))

        return current_layers, current_biases, delta_layers, delta_biases, train_error, test_error


    def layer_count(self):
        return len(self.layers)

    def __str__(self):
        result = 'layers:\n'
        for i in range(len(self.layers)):
            result += self.layers[i].__str__()
            result += '\n\n'
        result += 'biases:\n'
        for i in range(len(self.biases)):
            result += self.biases[i].__str__()
            result += '\n'

        return result
