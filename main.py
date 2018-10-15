import random
import datetime

from NeuralNetwork.Network import Network
from NeuralNetwork.NetworkSettings import NetworkSettings
from Helpers import *


def start():
    # Network settings
    print('Neural network settings started.\n')
    settings = NetworkSettings()

    # Data preparation
    ready_data = prepare_excel_data(settings)
    if settings.evaluate_learning_process == 1:
        random.shuffle(ready_data)
        split_idx = int(settings.split_for_eval_percent * len(ready_data))
        learning_set = ready_data[split_idx:]
        evaluation_set = ready_data[:split_idx]
    else:
        learning_set = ready_data
        evaluation_set = None

    # Neural Network
    NN = Network(settings)
    NN.start_teaching_process(training_data=learning_set,
                              epochs=settings.epochs,
                              mini_batch_size=settings.batch_size,
                              learning_rate=settings.learning_rate,
                              use_bias=settings.bias,
                              test_data=evaluation_set)

    result = NN.feedforward_batch(settings.testing_data, settings.problem_type)

    if settings.evaluate_learning_process:
       if settings.problem_type == 1:
            NN.error = [x * 100 for x in NN.error]  # error per batch in %
            show_plots(NN.error, result)
       else:
            show_plots(NN.error, result)

def show_plots(error_values, result):
    plt.ylabel('Error')
    plt.xlabel('Epoch number')
    plt.title('NN error for epoch')
    plt.plot(error_values)

    create_graph_for_data(np.asmatrix(np.array(result)))
    plt.ylabel('Y')
    plt.xlabel('X')

    plt.show()

if __name__ == "__main__":
    start()
