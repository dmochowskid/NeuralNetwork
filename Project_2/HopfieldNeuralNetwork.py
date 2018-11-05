import numpy as np
from random import randint, shuffle
from Helpers import *


class HopfieldNeuralNetwork:
    def __init__(self, image_size):
        self.weights = np.random.uniform(-1.0, 1.0, (image_size, image_size))
        self.num_inputs = image_size

    def calculate_neuron_output(self, neuron, input_pattern, original_value):
        num_neurons = len(input_pattern)
        s = 0.0
        for j in range(num_neurons):
            s += self.weights[neuron][j] * input_pattern[j]
        return 1.0 if s > 0.0 else (original_value if s == 0 else -1.0)

    def update(self, update_list, input_pattern, synchronous):
        result = input_pattern.copy()
        changed = False
        neuron_indexes_to_change = []
        for neuron_index in update_list:
            neuron_output = self.calculate_neuron_output(neuron_index, result, result[neuron_index])
            if neuron_output != result[neuron_index]:
                neuron_indexes_to_change.append((neuron_index, neuron_output))
        if len(neuron_indexes_to_change) > 0:
            if synchronous:
                for neruon_index in neuron_indexes_to_change:
                    result[neruon_index[0]] = neruon_index[1]
            else:
                neruon_index = neuron_indexes_to_change[random.randint(1, len(neuron_indexes_to_change)) - 1]
                result[neruon_index[0]] = neruon_index[1]
            changed = True
        return changed, result

    def test(self, input_pattern, synchronous):
        iteration_count = 0
        row_size = len(input_pattern)
        result = input_pattern.flatten().copy()
        while True:
            update_list = list(range(self.num_inputs))
            split_result = np.split(result, row_size)
            show_plot(split_result, iteration_count, False)
            changed, result = self.update(update_list, result, synchronous)
            iteration_count += 1
            if not changed:
                return split_result

    def calculate_weight(self, i, j, patterns):
        num_patterns = len(patterns)
        s = 0.0
        for k in range(num_patterns):
            s += patterns[k][i] * patterns[k][j]
        w = (1.0 / float(num_patterns)) * s
        return w

    def calculate_neuron_weights(self, neuron_index, input_patterns):
        weights = np.zeros(self.num_inputs)
        for j in range(self.num_inputs):
            if neuron_index == j: continue
            weights[j] = self.calculate_weight(neuron_index, j, input_patterns)
        return weights

    def train(self, input_patterns):
        weights = np.zeros((self.num_inputs, self.num_inputs))
        for i in range(self.num_inputs):
            weights[i] = self.calculate_neuron_weights(i, input_patterns)
        self.weights = weights
