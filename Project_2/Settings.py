class Settings:
    def __init__(self):
        self.input_data_path = ""
        self.row_size = int()

        self._prepare_demo_large_plus()
        self.neuron_size = 50

    def _prepare_demo_small(self):
        self.input_data_path = "./Data/small-7x7.csv"
        self.row_size = 7

    def _prepare_demo_large(self):
        self.input_data_path = "./Data/large-25x25.csv"
        self.row_size = 25

    def _prepare_demo_large_plus(self):
        self.input_data_path = "./Data/large-25x25.plus.csv"
        self.row_size = 25

    def _prepare_demo_animals(self):
        self.input_data_path = "./Data/animals-14x9.csv"
        self.row_size = 9

    def _prepare_demo_letters(self):
        self.input_data_path = "./Data/letters-8x12.csv"
        self.row_size = 12

    def _prepare_demo_letters_large(self):
        self.input_data_path = "./Data/letters-14x20.csv"
        self.row_size = 20

    def _prepare_demo_letters_abc(self):
        self.input_data_path = "./Data/letters-abc-8x12.csv"
        self.row_size = 12
