import numpy as np


class DataNormalizator:
    def __init__(self):
        pass

    # don't recommend to use in practice because of in a test set there may be more extreme elements
    # @staticmethod
    # def linear_normalize_into_unit(input_data):
    #     high = max(input_data)
    #     low = min(input_data)
    #     length = high - low
    #     f = lambda x: (x - low) / (high - low)
    #     return np.array(list(map(f, input_data)))

    # e.g. [-3, 7] -> [0, 1], then -13 -> -1 and 17 -> 2
    @staticmethod
    def linear_normalize_into_unit_with_range(input_data, start, end):
        length = end - start
        f = lambda x: (x - start) / (end - start)
        return np.array(list(map(f, input_data)))

    @staticmethod
    def linear_normalize_general(input_data, input_start, input_end, output_start, output_end):
        input_length = input_end - input_start
        output_length = output_end - output_start

        slope = (output_end - output_start) / (input_end - input_start)
        f = lambda x: output_start + slope * (x - input_start)
        return np.array(list(map(f, input_data)))

    @staticmethod
    def general_normalize(function, input_data):
        return np.array(list(map(function, input_data)))
