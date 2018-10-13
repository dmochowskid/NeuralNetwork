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
                              use_bias= settings.bias,
                              test_data=evaluation_set)

    result = NN.feedforward_batch(settings.testing_data, settings.problem_type)

    output_file = 'output' + str(datetime.datetime.now().isoformat().replace('.','-').replace(':', '-'))
    np.savetxt(output_file + '.csv', np.array(result), delimiter=',')
    settings.to_file(output_file + '.txt')

    if settings.evaluate_learning_process:
        if settings.problem_type == 1:
            output_file_error = 'error' + str(datetime.datetime.now().isoformat().replace('.', '-').replace(':', '-'))
            NN.error = [x * 100 for x in NN.error]  # error per batch in %
            plot_error(NN.error)
            np.savetxt(output_file_error + '.csv', np.array(NN.error), delimiter=',')
        else:
            output_file_error = 'error' + str(
                datetime.datetime.now().isoformat().replace('.', '-').replace(':', '-'))
            plot_error(NN.error)
            np.savetxt(output_file_error + '.csv', np.array(NN.error), delimiter=',')

def plot_error(error_values):
    plt.ylabel('Error')
    plt.xlabel('Epoch number')
    plt.title('NN error for epoch')
    plt.plot(error_values)
    axes = plt.gca()
    # axes.set_ylim([0, 110])
    plt.show()

if __name__ == "__main__":
    start()
