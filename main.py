import random
import datetime

from NeuralNetwork.Network import Network
from NeuralNetwork.NeuralNetworkConfiguration import NeuralNetworkConfiguration
from Helpers import *


def start():
    conf_type = input('To start neural network configuration press: [s]\n')
    # To load configuration from file press: [f]
    if conf_type == 's':
        configuration = start_configuration()
    elif conf_type == 'f':
        configuration = load_configuration_from_file()
    else:
        start()

    configuration.init_configuration()

    # Data prep
    ready_data = prepare_excel_data(configuration)

    if configuration.evaluate_learning_process == 1:
        random.shuffle(ready_data)
        split_idx = int(configuration.split_for_eval_percent * len(ready_data))
        learning_set = ready_data[split_idx:]
        evaluation_set = ready_data[:split_idx]
    else:
        learning_set = ready_data
        evaluation_set = None

    # Neural Network
    NN = Network(configuration)
    NN.start_teaching_process(training_data=learning_set,
                              epochs=configuration.epochs,
                              mini_batch_size=configuration.batch_size,
                              learning_rate=configuration.learning_rate,
                              use_bias= configuration.bias,
                              test_data=evaluation_set)

    result = NN.feedforward_batch(configuration.testing_data, configuration.problem_type)

    output_file = 'Output\\output' + str(datetime.datetime.now().isoformat().replace('.','-').replace(':', '-'))
    np.savetxt(output_file + '.csv', np.array(result), delimiter=',')
    configuration.to_file(output_file + '.txt')

    if configuration.evaluate_learning_process:
        if configuration.problem_type == 1:
            output_file_error = 'Output\\error' + str(datetime.datetime.now().isoformat().replace('.', '-').replace(':', '-'))
            NN.error = [x * 100 for x in NN.error]  # error per batch in %
            plot_error(NN.error)
            np.savetxt(output_file_error + '.csv', np.array(NN.error), delimiter=',')
        else:
            output_file_error = 'Output\\error' + str(
                datetime.datetime.now().isoformat().replace('.', '-').replace(':', '-'))
            plot_error(NN.error)
            np.savetxt(output_file_error + '.csv', np.array(NN.error), delimiter=',')

def start_configuration():
    configuration = NeuralNetworkConfiguration()
    configuration.problem_type = int(input('Problem type: [1] classification [2] regression '))
    configuration.layers = []

    if configuration.problem_type == 1:
        configuration.layers.append(2)
    else:
        configuration.layers.append(1)

    layers_count = int(input('Number of hidden layers:'))
    for x in range(1, layers_count + 1):
        neurons = int(input('Nuerons in layer #' + str(x) + ' '))
        configuration.layers.append(neurons)

    configuration.epochs = int(input('Epochs count: '))
    configuration.activation_function = int(input('Choose activation function: [1] polar, [2] bipolar '))
    configuration.learning_rate = float(input('Learning rate: '))
    configuration.batch_size = int(input('Batch size: '))
    configuration.inertia_ratio = float(input('Inertia ratio: '))
    configuration.evaluate_learning_process = int(input('Evaluate learning process? [1] yes [2] no '))
    configuration.learning_set_path = input('Enter learning data path: ')
    configuration.testing_set_path = input('Enter testing data path: ')

    print('Configuration has ended')
    return configuration


def plot_error(error_values):
    plt.ylabel('Error')
    plt.xlabel('Epoch number')
    plt.title('NN error for epoch')
    plt.plot(error_values)
    axes = plt.gca()
    # axes.set_ylim([0, 110])
    plt.show()


def load_configuration_from_file():
    return NeuralNetworkConfiguration()


def main():
    print('=============================')
    print('NUERAL NETWORKS: PROJECT 1')
    print('=============================')
    start()


if __name__ == "__main__":
    main()
