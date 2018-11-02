from Helpers import *
from Settings import *


def start():
    settings = Settings()
    images = prepare_input_data(settings.input_data_path, settings.row_size)

    for image in images:
        show_plot(image)


if __name__ == "__main__":
    start()
