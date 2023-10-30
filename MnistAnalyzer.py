import struct
import matplotlib.pyplot as plt
import seaborn
import numpy as np

from DataNormalizator import DataNormalizator
from MlpBase import MlpBase
from MlpSerializer import MlpSerializer
from functions import *


class MnistAnalyzer:
    def __init__(self):
        pass

    def get_labels(self, filename):
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
            for i in range(int(pictures_count)):
                result.append(unpacked[i * SIZE:i * SIZE + SIZE])

        # self.draw_number(result[1])
        return result

    def draw_number(self, pixels):
        side = 5
        image = []
        for i in range(28):
            image.append(pixels[i * 28:i * 28 + 28])

        plt.imshow(image, cmap=plt.get_cmap('gray'))
        plt.show()

    def normalize(self, labels, images):
        normalized_images = np.array(DataNormalizator.linear_normalize_into_unit_with_range(images, 0, 255))
        LOW_HOT = 0
        HIGH_HOT = 1

        normalized_labels = np.array(list(map(lambda x: self.one_hot_encode(LOW_HOT, HIGH_HOT, x), labels)))
        return normalized_labels, normalized_images

    def one_hot_encode(self, low, high, value):
        SIZE = 10
        encoded = np.full(10, low)
        encoded[value] = high
        return encoded

    def learn(self, seed, train_images, train_labels, test_images, test_labels, raw_test_labels):
        mlp = MlpBase(layers_description=[len(train_images[0]), 500, 100, 50, len(train_labels[0])],
                      _seed=seed,
                      activation=sigmoid,
                      activation_derivative=sigmoid_derivative,
                      last_layer_activation=linear,
                      last_layer_activation_derivative=linear_derivative,
                      loss=cross_entropy_with_softmax,
                      loss_gradient=cross_entropy_derivative_with_softmax,
                      descent_length=0.1
                      )

        for i in range(10000):
            indexes = np.random.choice(range(60000), 128)
            train_sample_images = train_images[indexes]
            train_sample_labels = train_labels[indexes]
            iteration_data = mlp.learn_iteration(train_sample_images, train_sample_labels, test_images[:100],
                                                 test_labels[:100])
            if i % 100 == 0:
                print("ITERATION {:5d}: TRAIN ERROR = {:8.5f}, TEST ERROR = {:8.5f}".format(
                    i + 1, iteration_data[4], iteration_data[5]))
                with open("TRAINERR.txt", "a") as outfile_train:
                    outfile_train.write(f"{iteration_data[4]},\n")
                with open("TESTERR.txt", "a") as outfile_test:
                    outfile_test.write(f"{iteration_data[5]},\n")

            if i % 100 == 0:
                right, wrong = self.perform_classification_test(mlp, test_images, raw_test_labels)
                print(f'right={right}, wrong={wrong} ({100 * right / (right + wrong)}%)')
                with open("ACCURACY.txt", "a") as outfile_accuracy:
                    outfile_accuracy.write(f"{100 * right / (right + wrong)},\n")
                if i == 4200:
                    ms = MlpSerializer()
                    ms.serialize(mlp, "97rated-network-4200")
                if 100 * right / (right + wrong) >= 97.5:
                    ms = MlpSerializer()
                    ms.serialize(mlp, f"975rated-network-{i}")
                    return mlp

        return mlp

    def do_all(self, seed):
        print('reading mnist files...')
        raw_labels = self.get_labels('mnist/train-labels.idx1-ubyte')
        raw_images = self.get_images('mnist/train-images.idx3-ubyte')
        raw_test_labels = self.get_labels('mnist/t10k-labels.idx1-ubyte')
        raw_test_images = self.get_images('mnist/t10k-images.idx3-ubyte')
        n_labels, n_images = self.normalize(np.array(raw_labels), np.array(raw_images))
        n_test_labels, n_test_images = self.normalize(np.array(raw_test_labels), np.array(raw_test_images))

        print('learning...')
        nn = self.learn(seed, n_images, n_labels, n_test_images, n_test_labels, raw_test_labels)
        # right, wrong = self.perform_classification_test(nn, n_test_images, raw_test_labels)
        # print(f'right={right}, wrong={wrong} ({100*right / (right + wrong)}%)')

    def perform_classification_test(self, nn, n_images, raw_labels):
        right = 0
        wrong = 0

        for i, image in enumerate(n_images):
            output = nn.operation(image)
            answer = self.onehot_to_value(output)
            if answer == raw_labels[i]:
                right += 1
            else:
                wrong += 1
        return right, wrong

    def onehot_to_value(self, one_hot_encoded):
        return one_hot_encoded.argmax()

    def check_network(self, nn):
        print('reading mnist files...')

        raw_test_labels = self.get_labels('mnist/t10k-labels.idx1-ubyte')
        raw_test_images = self.get_images('mnist/t10k-images.idx3-ubyte')
        n_test_labels, n_test_images = self.normalize(np.array(raw_test_labels), np.array(raw_test_images))
        right, wrong = self.perform_classification_test(nn, n_test_images, raw_test_labels)
        print(f'right={right}, wrong={wrong} ({100 * right / (right + wrong)}%)')

    def draw_confusion_matrix(self, nn):
        print('reading mnist files...')

        raw_test_labels = self.get_labels('mnist/t10k-labels.idx1-ubyte')
        raw_test_images = self.get_images('mnist/t10k-images.idx3-ubyte')
        n_test_labels, n_test_images = self.normalize(np.array(raw_test_labels), np.array(raw_test_images))
        guesses = np.array([np.zeros(10) for i in range(10)])

        for i, image in enumerate(n_test_images):
            output = nn.operation(image)
            answer = self.onehot_to_value(output)
            guesses[answer][raw_test_labels[i]] += 1

        ax = seaborn.heatmap(guesses, xticklabels='0123456789', yticklabels='123456789', annot=True, square=True,
                             fmt='g')
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        plt.show()

    def draw_plot(self):
        accuracy = [28.63,
                    87.75,
                    87.24,
                    92.74,
                    92.34,
                    93.0,
                    93.77,
                    89.05,
                    92.64,
                    94.88,
                    94.64,
                    94.84,
                    95.04,
                    95.32,
                    95.35,
                    95.16,
                    93.88,
                    95.61,
                    96.01,
                    96.2,
                    95.7,
                    95.91,
                    96.13,
                    96.58,
                    96.39,
                    96.64,
                    96.28,
                    96.07,
                    96.12,
                    96.76,
                    96.46,
                    96.46,
                    96.51,
                    96.37,
                    96.3,
                    96.77,
                    96.49,
                    96.18,
                    96.89,
                    96.82,
                    96.76,
                    96.36,
                    97.04,
                    96.72,
                    96.98,
                    96.85,
                    96.96,
                    96.9,
                    96.96,
                    96.73,
                    96.81,
                    96.94,
                    97.25,
                    97.19,
                    96.93,
                    97.08,
                    96.99,
                    96.92,
                    97.11,
                    96.92,
                    96.53,
                    96.94,
                    97.11,
                    97.1,
                    96.34,
                    96.81,
                    97.2,
                    96.7,
                    97.35,
                    96.56,
                    97.23,
                    97.27,
                    97.27,
                    97.06,
                    97.07,
                    97.04,
                    97.11,
                    96.87,
                    97.03,
                    97.16,
                    97.35,
                    97.33,
                    96.98,
                    96.56,
                    97.3,
                    97.3,
                    97.36,
                    97.16,
                    97.46,
                    97.42,
                    97.09,
                    97.34,
                    97.36,
                    97.6,
                    ]
        train = [2.626845214286431,
                 0.4260421123109079,
                 0.24326241362843576,
                 0.2171013384268503,
                 0.3273933202051838,
                 0.21240254896567495,
                 0.15803274367449446,
                 0.20372584802978344,
                 0.3994313974393842,
                 0.2341677227804017,
                 0.14778597856928824,
                 0.06012083722438984,
                 0.12582612252155112,
                 0.0733741462506839,
                 0.10561695679914584,
                 0.06290325585973877,
                 0.2288593931787025,
                 0.09988764264245635,
                 0.05936949092535318,
                 0.026362926760440258,
                 0.18738392077337163,
                 0.05415745795203785,
                 0.15596699272391776,
                 0.011487173659376793,
                 0.029233177867697805,
                 0.07289624156988186,
                 0.04182476862629199,
                 0.07971170050894663,
                 0.05485879800030667,
                 0.022344749366206448,
                 0.037793715602410335,
                 0.08202430974830002,
                 0.03450519152274971,
                 0.11141027623661634,
                 0.186721513333969,
                 0.06355778241134433,
                 0.04990654341101205,
                 0.05993835068051435,
                 0.06935372289208161,
                 0.06397723256507874,
                 0.019496645912986037,
                 0.10011903751765988,
                 0.04909090291049497,
                 0.06941641026573445,
                 0.07157144700812679,
                 0.02491066348205271,
                 0.055709194737670484,
                 0.0023877025609838343,
                 0.03178239498853053,
                 0.028527925085694284,
                 0.03275815873323359,
                 0.020742650804977933,
                 0.06605740992740013,
                 0.004065225718179441,
                 0.027427205833948653,
                 0.02320992068973326,
                 0.017175349807938355,
                 0.05045630297331121,
                 0.03713546264586127,
                 0.0396676996032932,
                 0.0346246767425738,
                 0.021315990585306376,
                 0.08351612548452478,
                 0.15768602949997407,
                 0.04818690364701919,
                 0.12515796730244147,
                 0.009282218655284044,
                 0.01530273478642754,
                 0.01699358788134372,
                 0.057427562100774224,
                 0.04846690832677241,
                 0.028156488045142408,
                 0.013450288657924471,
                 0.043700060127232994,
                 0.008968671456584587,
                 0.04035909695850992,
                 0.03214578255102183,
                 0.04271239430470554,
                 0.00143545042667216,
                 0.03595455191366702,
                 0.003750204166209079,
                 0.031337735029383494,
                 0.033826882173865924,
                 0.028193327688034397,
                 0.018277031748275052,
                 0.005589220018791253,
                 0.0072956818820261796,
                 0.014294103077212813,
                 0.00188986889116728,
                 0.08669837043499178,
                 0.012642208836383754,
                 0.0452227916161598,
                 0.012322348088320362,
                 0.019449988636905582,
                 ]
        test = [1.9566843190411398,
                0.267431664913321,
                0.2938379868755735,
                0.16979558370134715,
                0.20072812171962984,
                0.13872144390522137,
                0.11607980458503557,
                0.3567738926296272,
                0.17179895855530375,
                0.09581204957681053,
                0.10788095647326358,
                0.12704017618299468,
                0.08758574110541366,
                0.11477761333525555,
                0.05747226862868628,
                0.04641890331387532,
                0.10222351431041439,
                0.05342023199694013,
                0.060062703658061045,
                0.04025122257630193,
                0.1450652892156323,
                0.07082257089362815,
                0.09687577229560299,
                0.09442549940054805,
                0.033173286811433784,
                0.06550211774771307,
                0.04925720921701541,
                0.0436798004692794,
                0.046219340646737994,
                0.06369473263382791,
                0.05113198057627761,
                0.037207974758162086,
                0.040511706113399396,
                0.08534078276543665,
                0.044266839630125156,
                0.06662616964900318,
                0.046109339841356796,
                0.026511569319994332,
                0.03958468742855815,
                0.022130733937251776,
                0.0690083668394608,
                0.05042427355827584,
                0.06331959776507716,
                0.024163800635148548,
                0.01784596859078403,
                0.025267767404827918,
                0.029998583406081872,
                0.01866361097395079,
                0.033907850398014285,
                0.02955666968278174,
                0.016833916547063572,
                0.029292410991770903,
                0.04285769296999557,
                0.06558290855862141,
                0.035088304246585406,
                0.022862272666807196,
                0.030916323791277147,
                0.04041054151189358,
                0.04369976069002575,
                0.100081812742898,
                0.14748545631054333,
                0.11631785172806375,
                0.09297903332223477,
                0.12182287767671078,
                0.037865570918664444,
                0.07376651330156628,
                0.046606296936987805,
                0.039471366335662124,
                0.01610484518221637,
                0.04930364247513183,
                0.05110379879549406,
                0.02356809054131797,
                0.0295931569834871,
                0.022482313775009994,
                0.06299477810614955,
                0.022285160645149467,
                0.02052116010356252,
                0.047580106451478324,
                0.02372674264694639,
                0.012273541373938414,
                0.044970396487437364,
                0.007170756835403482,
                0.06251503653100637,
                0.2056046262721312,
                0.01156658192383645,
                0.006344120407841717,
                0.010220404968731693,
                0.014792557830094118,
                0.06418859943784418,
                0.007197193772480657,
                0.03670017164169301,
                0.007874662207487053,
                0.0221869091114412,
                0.011631263942917268,
                ]
        x_axis = [i * 100 for i in range(94)]
        plt.plot(x_axis[1:], accuracy[1:], label = "Dokładność")
        plt.xlabel("Liczba iteracji")
        plt.ylabel("%")
        plt.legend()
        plt.show()

# (500 - 100 - 50)
# iteration 640 94%+
# iteration 1040 95%+
# iteration 1800 96%+
# iteration 4200 97%+
