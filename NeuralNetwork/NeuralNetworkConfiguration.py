import numpy as np
import csv

from Helpers import get_column


class NeuralNetworkConfiguration():
    # default settings
    def __init__(self):
        self.layers = [2, 6 ]
        self.activation_function = 2
        self.bias = True
        self.learning_rate = 0.1
        self.inertia_ratio = 0.2
        self.problem_type = 1  # 1 - classification, 2 - regression
        self.epochs = 100
        self.batch_size = 10
        self.evaluate_learning_process = True  # true=use validation set
        # Input
        self.split_for_eval_percent = 0.1  # if verbose marked split 10% of input data for "evaluation after each epoch"
        self.learning_set_path = 'Data\Classification\data.circles.train.10000.csv'
        self.testing_set_path = 'Data\Classification\data.circles.test.10000.csv'
        self.learning_data = []
        self.testing_data = []

    def init_configuration(self):
        print('Start loading...\n')

        data = np.loadtxt(self.learning_set_path, delimiter=',', skiprows=1)
        self.learning_data = data

        if(self.problem_type == 1):
            column = get_column(self.learning_data, 2)
            classes_count = len(np.unique(column))
            self.layers.append(classes_count)
        else:
            self.layers.append(1)

        data = np.loadtxt(self.testing_set_path, delimiter=',', skiprows=1)

        for row in data:
            if hasattr(row, '__iter__'):
                self.testing_data.append(row[:self.layers[0]])  # take only as much input as can be processed by the NN
            else:
                self.testing_data.append(row)

        return

    def to_file(self, file_path):
        additional_info = {
            'problem_type': '#1 - classification, 2 - regression\n',
            'activation_function': '#1 - polar (logistic function), 2 - bipolar (arctan)\n'
        }

        f = open(file_path, 'w')
        props = vars(self)
        for prop in props:
            prop_str = str(prop)
            f.write(additional_info.get(prop_str, ''))
            if prop_str != 'learning_data' and prop_str != 'testing_data':
                f.write(prop_str + ' : ' + str(props[prop]))
                f.write('\n')
        f.close()
