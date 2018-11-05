from Helpers import *
from Settings import *
from HopfieldNeuralNetwork import *


def start():
    settings = Settings()
    images = prepare_input_data(settings.input_data_path, settings.row_size)

    for image in images:
        show_plot(image, "Example")

    # training
    hopfield_neural_network = HopfieldNeuralNetwork(settings.row_size * settings.column_size)
    flatten_images = [np.array(image).flatten() for image in images]
    hopfield_neural_network.train(flatten_images)

    # test
    noised_image = add_noise(images[settings.test_image_index], settings.noise)
    result_image = hopfield_neural_network.test(np.array(noised_image), settings.synchronous)
    show_plot(result_image, "Result", False)


if __name__ == "__main__":
    start()
