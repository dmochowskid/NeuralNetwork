import numpy as np
import matplotlib.pyplot as plt
import random


def prepare_input_data(input_data_path, row_size):
    """Return the list of images in two dimensions"""
    input_data = np.loadtxt(input_data_path, delimiter=',')
    images = []
    for row in input_data:
        images.append(np.split(row, row_size))
    return images


def two_images_are_equal(first, second):
    return first.equal(second)


def get_random_image(column_size, row_size):
    random_image = np.empty([column_size, row_size], dtype=int)
    for column in random_image:
        for i in range(len(column)):
            column[i] = 1 if random.random() > 0.5 else -1
    return random_image


def show_plot(image, title='Image', color_reversed=True):
    """image - image in two dimensions"""
    image_copy = image.copy()
    if color_reversed:
        for i in range(len(image)):
            for j in range(len(image[0])):
                image_copy[i][j] = -1 if image[i][j] == 1 else 1
    plt.ylabel('Y')
    plt.xlabel('X')
    plt.title(title)
    plt.imshow(image_copy)
    plt.show()
