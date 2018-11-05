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
        """Run the network using the input data until the output state doesn't change
        or a maximum number of iteration has been reached."""
        iteration_count = 0
        num_images = len(input_pattern)
        input_pattern = input_pattern.flatten()
        result = input_pattern.copy()
        while True:
            update_list = list(range(self.num_inputs))
            shuffle(update_list)
            split_result = np.split(result, num_images)
            show_plot(split_result, iteration_count, False)
            changed, result = self.run_once(update_list, result)
            iteration_count += 1
            if not changed:
                return split_result

    def calculate_weight(self, i, j, patterns):
        """Calculate the weight between the given neurons"""
        num_patterns = len(patterns)
        s = 0.0
        for mu in range(num_patterns):
            s += patterns[mu][i] * patterns[mu][j]
        w = (1.0 / float(num_patterns)) * s
        return w

    def calculate_neuron_weights(self, neuron_index, input_patterns):
        """Calculate the weights for the givven neuron"""
        num_neurons = len(input_patterns[0])
        weights = np.zeros(num_neurons)
        for j in range(num_neurons):
            if neuron_index == j: continue
            weights[j] = self.calculate_weight(neuron_index, j, input_patterns)
        return weights

    def hebbian_training(self, input_patterns):
        """Train a network using the Hebbian learning rule"""
        num_neurons = len(input_patterns[0])
        weights = np.zeros((num_neurons, num_neurons))
        for i in range(num_neurons):
            weights[i] = self.calculate_neuron_weights(i, input_patterns)
        self.weights = weights
