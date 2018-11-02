import numpy as np
import matplotlib.pyplot as plt


def prepare_input_data(input_data_path, row_size):
    """Return the list of images in two dimensions"""
    input_data = np.loadtxt(input_data_path, delimiter=',')
    images = []
    for row in input_data:
        images.append(np.split(row, row_size))
    return images


def show_plot(image):
    """image - image in two dimensions"""
    plt.ylabel('Y')
    plt.xlabel('X')
    plt.title('Image')
    plt.imshow(image)
    plt.show()
