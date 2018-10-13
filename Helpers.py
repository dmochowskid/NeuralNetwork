import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cbook as cbook


def prepare_excel_data(configuration):
    # have to be passed as tuple<input, output>
    # so the last number is a class
    if configuration.problem_type == 1:
        input_size = configuration.layers[0]  # number of inputs
        output_size = configuration.layers[-1]  # number of classes
        ready_data = []
        for row in configuration.learning_data:
            output = np.zeros((output_size,), dtype=np.int)
            # take last number in a row as a class and set it's id
            # which corresponds to output neuron to value 1 and rest to 0
            class_id = int(row[-1])
            output[class_id - 1] = 1
            ready_data.append(((row[:input_size])[None].T, output[None].T))
        return ready_data
    if configuration.problem_type == 2:
        input_size = configuration.layers[0]  # number of inputs
        ready_data = []
        for row in configuration.learning_data:
            output = np.zeros((1,), dtype=np.double)
            # take last number in a row as a class and set it's id
            # which corresponds to output neuron to value 1 and rest to 0
            output[0] = row[input_size]  # result
            ready_data.append(((row[:input_size])[None].T, output[None].T))
        return ready_data


def get_column(arr, col):
    return np.array(arr[:, col].T)


def create_graph_for_data(data):
    dim = np.shape(data)[1]
    if isinstance(data, list):
        print('its error')
        dim = 1

    if dim == 3:
        x = get_column(data, 0)
        y = get_column(data, 1)
        s = get_column(data, 2)
        fig, ax = plt.subplots()
        ax.scatter(x=x, y=y, s=np.ones(len(x)), c=s)
        ax.grid(True)
        fig.tight_layout()
        return ax

    if dim == 2:
        sorted_data = np.sort(data, axis=0)
        x = get_column(sorted_data, 0)
        y = get_column(sorted_data, 1)
        fig, ax = plt.subplots()
        ax.scatter(x=x, y=y)
        ax.grid(True)
        fig.tight_layout()
        return ax
