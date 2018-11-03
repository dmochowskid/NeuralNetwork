from Helpers import *
from Settings import *
from HopfieldNeuralNetwork import *


def start():
    settings = Settings()
    images = prepare_input_data(settings.input_data_path, settings.row_size)

    for image in images:
        show_plot(image)

    # training
    hopfield_neural_network = HopfieldNeuralNetwork()
    hopfield_neural_network.prepare(settings.row_size * settings.column_size)
    flatten_images = [np.array(image).flatten() for image in images]
    hopfield_neural_network.hebbian_training(flatten_images)

    # test
    random_image = get_random_image(settings.column_size, settings.row_size)
    result_image = hopfield_neural_network.run(random_image.flatten())
    show_plot(result_image)


if __name__ == "__main__":
    start()
