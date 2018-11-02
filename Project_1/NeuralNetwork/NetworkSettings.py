import numpy as np
import csv

from Helpers import get_column


class NetworkSettings():
    def __init__(self):
        self.layers = [1, 50, 50, 50, 50, 50]
        self.activation_function = 2
        self.bias = True
        self.learning_rate = 0.02
        self.inertia_ratio = 0.5
        self.problem_type = 2 # 1 - classification, 2 - regression
        self.epochs = 100
        self.batch_size = 10
        self.evaluate_learning_process = True  # true=use validation set
        # Input
        self.split_for_eval_percent = 0.1  # if verbose marked split 10% of input data for "evaluation after each epoch"
        self.learning_set_path = 'SN_projekt1_test/Regression/data.multimodal.train.1000.csv'
        self.testing_set_path = 'SN_projekt1_test/Regression/data.multimodal.test.1000.csv'
        #self.learning_set_path = 'SN_projekt1_test/Classification/data.XOR.train.1000.csv'
            #self.testing_set_path = 'SN_projekt1_test/Classification/data.XOR.test.1000.csv'
            data = np.loadtxt(self.learning_set_path, delimiter=',', skiprows=1)
        self.learning_data = data

        if(self.problem_type == 1):
            column = get_column(self.learning_data, 2)
            classes_count = len(np.unique(column))
            self.layers.append(classes_count)
        else:
            self.layers.append(1)

        data = np.loadtxt(self.testing_set_path, delimiter=',', skiprows=1)
        self.testing_data = []
        for row in data:
            self.testing_data.append(row)

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
