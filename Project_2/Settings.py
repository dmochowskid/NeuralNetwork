class Settings:
    def __init__(self):
        self.input_data_path = ""
        self.row_size = int()
        self.column_size = int()
        self.use_own_test = False

        self._prepare_demo_oscillation()
        self.synchronous = True
        self.noise = 0.30
        self.test_image_index = 1


    def _prepare_demo_small(self):
        self.input_data_path = "./Data/small-7x7.csv"
        self.row_size = 7
        self.column_size = 7

    def _prepare_demo_large(self):
        self.input_data_path = "./Data/large-25x25.csv"
        self.row_size = 25
        self.column_size = 25

    def _prepare_demo_large_plus(self):
        self.input_data_path = "./Data/large-25x25.plus.csv"
        self.row_size = 25
        self.column_size = 25

    def _prepare_demo_animals(self):
        self.input_data_path = "./Data/animals-14x9.csv"
        self.row_size = 9
        self.column_size = 14

    def _prepare_demo_letters(self):
        self.input_data_path = "./Data/letters-8x12.csv"
        self.row_size = 12
        self.column_size = 8

    def _prepare_demo_letters_large(self):
        self.input_data_path = "./Data/letters-14x20.csv"
        self.row_size = 20
        self.column_size = 14

    def _prepare_demo_letters_abc(self):
        self.input_data_path = "./Data/letters-abc-8x12.csv"
        self.row_size = 12
        self.column_size = 8

    def _prepare_demo_cats(self):
        self.input_data_path = "./Data/cats-50x100.csv"
        self.row_size = 50
        self.column_size = 100

    def _prepare_demo_oscillation(self):
        self.input_data_path = "./Data/oscillation.csv"
        self.row_size = 2
        self.column_size = 7
        self.use_own_test = True
        self.own_test = [[1, 1, 1, 1, 1, 1, 1],
                         [-1, -1, -1, -1, -1, -1, -1]]
