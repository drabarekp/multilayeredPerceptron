import struct
import matplotlib.pyplot as plt
import numpy as np

class MnistAnalyzer:
    def __init__(self):
        pass

    @staticmethod
    def get_labels(filename):
        with open(filename, mode='rb') as file:
            file_content = file.read()
            unpacked = struct.unpack("B" * (len(file_content) - 8), file_content[8:])
            return unpacked

    def get_images(self, filename):
        SIZE = 28 * 28
        with open(filename, mode='rb') as file:
            file_content = file.read()
            unpacked = struct.unpack("B" * (len(file_content) - 16), file_content[16:])
            pictures_count = len(unpacked) / SIZE
            result = []
            for i in range(SIZE):
                result.append(unpacked[i * SIZE:i * SIZE + SIZE])

        self.draw_number(result[1])
        return result

    def draw_number(self, pixels):
        side = 5
        image = []
        for i in range(28):
            image.append(pixels[i * 28:i * 28 + 28])

        plt.imshow(image, cmap=plt.get_cmap('gray'))
        plt.show()