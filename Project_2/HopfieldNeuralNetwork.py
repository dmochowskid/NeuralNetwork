import numpy as np
from random import randint, shuffle
from Helpers import *


class HopfieldNeuralNetwork:
    def __init__(self, image_size):
        self.weights = np.random.uniform(-1.0, 1.0, (image_size, image_size))
        self.num_inputs = image_size

    def calculate_neuron_output(self, neuron, input_pattern):
        """Calculate the output of the given neuron"""
        num_neurons = len(input_pattern)
        s = 0.0
        for j in range(num_neurons):
            s += self.weights[neuron][j] * input_pattern[j]
        return 1.0 if s > 0.0 else -1.0

    def run_once(self, update_list, input_pattern):
        """Iterate over every neuron and update it's output"""
        result = input_pattern.copy()
        changed = False
        for neuron in update_list:
            neuron_output = self.calculate_neuron_output(neuron, result)
            if neuron_output != result[neuron]:
                result[neuron] = neuron_output
                changed = True
        return changed, result

    def run(self, input_pattern):
        """Run the network using the input data until the output state doesn't change"""
        iteration_count = 0
        row_size = len(input_pattern)
        result = input_pattern.flatten().copy()
        while True:
            update_list = list(range(self.num_inputs))
            split_result = np.split(result, row_size)
            show_plot(split_result, iteration_count, False)
            changed, result = self.run_once(update_list, result)
            iteration_count += 1
            if not changed:
                return split_result

    def calculate_weight(self, i, j, patterns):
        """Calculate the weight between the given neurons"""
        num_patterns = len(patterns)
        s = 0.0
        for k in range(num_patterns):
            s += patterns[k][i] * patterns[k][j]
        w = (1.0 / float(num_patterns)) * s
        return w

    def calculate_neuron_weights(self, neuron_index, input_patterns):
        """Calculate the weights for the givven neuron"""
        weights = np.zeros(self.num_inputs)
        for j in range(self.num_inputs):
            if neuron_index == j: continue
            weights[j] = self.calculate_weight(neuron_index, j, input_patterns)
        return weights

    def hebbian_training(self, input_patterns):
        weights = np.zeros((self.num_inputs, self.num_inputs))
        for i in range(self.num_inputs):
            weights[i] = self.calculate_neuron_weights(i, input_patterns)
        self.weights = weights
