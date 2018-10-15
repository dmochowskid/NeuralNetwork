import sys, csv, numpy as np

from numpy import exp as e

from NeuralNetwork.NetworkSettings import NetworkSettings

layerscount = 2;
weightM = []
neurons = []
activationFun = 1
input = []


class Network(object):
    layers_count = 0
    layers = []
    biases = []
    weights = []
    velocities = []
    activation_fun = 1

    def __init__(self, config):
        self.layers_count = len(config.layers)
        self.layers = config.layers
        self.problem_type = config.problem_type
        self.error = []
        # init biases, weights and velocities with random data with normal distribution
        self.biases = [np.random.randn(y, 1) for y in self.layers[1:]]
        self.velocities = [np.random.randn(y, x)
                           for x, y in zip(self.layers[:-1], self.layers[1:])]

        self.weights = [np.random.randn(y, x)
                        for x, y in zip(self.layers[:-1], self.layers[1:])]

        self.activation_fun = config.activation_function

    def feedforward_batch(self, data, problem_type):
        result = []
        for sample in data:
            if problem_type == 1:
                point = sample[:2]
                ret = self.feedforward(np.asmatrix((point)).T)
                result.append(np.array([sample[0], point[1], np.argmax(ret) + 1]))
            elif problem_type == 2:
                point = sample[:1]
                ret = self.feedforward(np.asmatrix((point)).T)
                if isinstance(point, list):
                    result.append(np.array([point[0], ret]))
                else:
                    result.append(np.array([point, ret]))
        return result

    def feedforward_batch_with_only_result(self, data, problem_type):
        result = []
        for sample in data:
            ret = self.feedforward(np.asmatrix((sample)).T)
            if problem_type == 1:
                result.append(np.array(np.argmax(ret) + 1))
            else:
                result.append(np.array(ret))
        return result

    def feedforward(self, a):
        i = 0;
        for b, w in zip(self.biases, self.weights):
            i += 1
            z = np.dot(w, a) + b
            if i == len(self.weights) and self.problem_type == 2:
                a = z
            else:
                a = sigmoid(z, self.activation_fun)
        return a

    # start training using stochastic gradient descend
    def start_teaching_process(self,
                               training_data,
                               epochs,
                               mini_batch_size,
                               learning_rate,
                               use_bias=True,
                               test_data=None,
                               inertia_ratio=0):
        if test_data:
            test_data_length = len(test_data)

        learning_data_length = len(training_data)
        for j in range(epochs):
            # shuffle data
            np.random.shuffle(training_data)

            # divide training data into separated batches of size `mini_batch_size`
            mini_batches = [
                training_data[k:k + mini_batch_size]
                for k in range(0, learning_data_length, mini_batch_size)]

            # for each batch separately do learning process
            for mini_batch in mini_batches:
                self._update_mini_batch(mini_batch, learning_rate, use_bias, inertia_ratio)

            # if some validation data was set, check how network is performing
            if test_data:
                if self.problem_type == 1:
                    correct_answers = self._evaluate_learning_process(test_data)
                    print("Epoch {0}: {1} / {2}".format(j, correct_answers, test_data_length))
                    self.error.append(1 - (correct_answers/test_data_length))  # error in %
                elif self.problem_type == 2:
                    results = self._evaluate_learning_process(test_data)
                    errors = [(x[0]-y[0])[0] for (x, y) in results]
                    average_error = sum(errors) / len(errors)
                    self.error.append(average_error)
                    print("Epoch {0}: average error {1}".format(j, average_error))

            else:
                print("Epoch {0} complete".format(j))

    def _update_mini_batch(self, mini_batch, learning_rate, use_bias, inertia_ratio):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        # for data in batch perform backpropagation algorithm
        for x, y in mini_batch:
            delta_nabla_b, delta_nabla_w = self._do_backpropagation(x, y, use_bias)
            nabla_b = [nb + dnb for nb, dnb in zip(nabla_b, delta_nabla_b)]
            nabla_w = [nw + dnw for nw, dnw in zip(nabla_w, delta_nabla_w)]

        self.velocities = [inertia_ratio * v - (learning_rate / len(mini_batch)) * nw
                           for v, nw in zip(self.velocities, nabla_w)]

        self.weights = [w + v_prime
                        for w, v_prime in zip(self.weights, self.velocities)]

        self.biases = [b - (learning_rate / len(mini_batch)) * nb
                       for b, nb in zip(self.biases, nabla_b)]


    def _do_backpropagation(self, x, y, use_bias):
        nabla_b = [np.zeros(b.shape) for b in self.biases]
        nabla_w = [np.zeros(w.shape) for w in self.weights]

        activation = x
        activations = [x]
        zs = []

        i = 0
        for b, w in zip(self.biases, self.weights):
            i += 1
            if use_bias:
                z = np.dot(w, activation) + b
            else:
                z = np.dot(w, activation)

            zs.append(z)

            #use linear function if it is regression, sigmoid otherwise
            if i == len(self.weights) and self.problem_type == 2:
                activation = z
            else:
                activation = sigmoid(z, self.activation_fun)

            activations.append(activation)

        if self.problem_type == 2:
            activation_fun_prime = 1  # derivative for linear fun
        elif self.problem_type == 1:
            activation_fun_prime = sigmoid_prime(zs[-1], self.activation_fun)

        delta = self.cost_derivative(activations[-1], y) * activation_fun_prime
        nabla_b[-1] = delta
        nabla_w[-1] = np.dot(delta, activations[-2].transpose())

        for layer_idx in range(2, self.layers_count):
            z = zs[-layer_idx]

            sp = sigmoid_prime(z, self.activation_fun)

            delta = np.dot(self.weights[-layer_idx + 1].transpose(), delta) * sp
            nabla_b[-layer_idx] = delta
            nabla_w[-layer_idx] = np.dot(delta, activations[-layer_idx - 1].transpose())

        return (nabla_b, nabla_w)

    def _evaluate_learning_process(self, test_data):
        if self.problem_type == 1:
            test_results = [(np.argmax(self.feedforward(x)), y)
                            for (x, y) in test_data]

            correct_ans_count = 0
            for x, y in test_results:
                if y[x] == 1:
                    correct_ans_count += 1

            return correct_ans_count
        elif self.problem_type == 2:
            return [(self.feedforward(x), y) for (x, y) in test_data]

    def cost_derivative(self, output_activations, y):
        return output_activations - y


def sigmoid(z, function_type=1):
    if function_type == 1:
        return 1.0 / (1.0 + e(-z))  # logistic function
    else:
        return np.arctan(z)


def sigmoid_prime(z, function_type=1):
    # Derivative of the sigmoid function.
    if function_type == 1:
        return (sigmoid(z, function_type) * (1 - sigmoid(z, function_type)))
    else:
        return 1 / (1 + z ** 2)  # arctan derivative