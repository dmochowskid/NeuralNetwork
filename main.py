import random
import datetime
from matplotlib.colors import ListedColormap

from NeuralNetwork.Network import Network
from NeuralNetwork.NetworkSettings import NetworkSettings
from Helpers import *

cmap_light = ListedColormap(['#8716b7', '#2bc6bf', '#c4b215'])

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
            show_plots(settings, NN, result)
       else:
            show_plots(settings, NN, result)

def show_plots(settings, NN, result):
    plt.ylabel('Error')
    plt.xlabel('Epoch number')
    plt.title('NN error for epoch')
    plt.plot(NN.error)

    plt.figure()

    if settings.problem_type == 1:
        h = .01
        x_min, x_max = -1, 1
        y_min, y_max = -1, 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        mesh_points = np.c_[xx.ravel(), yy.ravel()]

        z = np.array(NN.feedforward_batch_with_only_result(mesh_points, settings.problem_type))
        z = z.reshape(xx.shape)

        x = []
        y = []
        s = []
        for e in settings.testing_data:
            x.append(e[0])
            y.append(e[1])
            s.append(e[2])
        plt.pcolormesh(xx, yy, z, cmap=cmap_light)
        plt.scatter(x=x, y=y, s=np.ones(len(x)), c=s)

        plt.xlim(x_min, x_max)
        plt.ylim(y_min, y_max)

        # create_graph_for_data(np.asmatrix(np.array(result)))
        plt.ylabel('Y')
        plt.xlabel('X')
    else:
        x = []
        y = []
        for e in settings.testing_data:
            x.append(e[0])
            y.append(e[1])

        y_own = np.asmatrix(np.array(result))
        sorted_data = np.sort(y_own, axis=0)
        x2 = get_column(sorted_data, 0)[0].tolist()
        y2 = get_column(sorted_data, 1)[0].tolist()

        plt.plot(x , y)
        plt.plot(x2, y2)
        plt.legend(['Test', 'Estimate'], loc='upper left')
        plt.grid(True)
        plt.ylabel('Y')
        plt.xlabel('X')

    plt.show()

if __name__ == "__main__":
    start()
