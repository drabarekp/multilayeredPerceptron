import numpy as np

from dash import Dash, dcc, html, Input, Output
from MlpBase import MlpBase
from DataNormalizator import DataNormalizator
from MlpSerializer import MlpSerializer
from figure import get_figure
from functions import *
from read_file import read_classification, read_regression, normalize_regression
from MnistAnalyzer import MnistAnalyzer

np.seterr(all='raise')
ITER = 100
LAYERS = [1, 4, 4, 1]
SEED = 1666

app = Dash('Network plot')
DATA = []   # data must be accessible for automatic calls


@app.callback(Output("graph", "figure"),
              Input('slider', 'value'))
def select_iteration(number):
    return get_figure(DATA[number - 1], LAYERS)


def peek(data_input, data_output, test_input, test_output):
    mlp = MlpBase(layers_description=LAYERS,
                  _seed=SEED,
                  activation=sigmoid,
                  activation_derivative=sigmoid_derivative,
                  last_layer_activation=sigmoid,
                  last_layer_activation_derivative=sigmoid_derivative,
                  loss=mean_squared_error,
                  loss_gradient=mean_squared_error_derivative,
                  descent_length=0.1
                  )

    iterations = []
    for i in range(ITER):
        iteration_data = mlp.learn_iteration(data_input, data_output, test_input, test_output)
        iterations.append(iteration_data)

        if i % 10 == 9:
            print("ITERATION {:5d}: TRAIN ERROR = {:8.3f}, TEST ERROR = {:8.3f}".format(
                i + 1, iteration_data[4], iteration_data[5]))

    global DATA
    DATA = iterations

if __name__ == '__main__':

    mnist = MnistAnalyzer()
    mnist.draw_plot()
    #mser = MlpSerializer()
    #nn = mser.deserialize('97rated-network-4200')
    #mnist.draw_confusion_matrix(nn)
    #mnist.do_all(SEED)


    # read_classification('data_classification/data.three_gauss.train.100.csv')
    #
    # train_in, train_out = read_regression('data_regression/data.activation.train.100.csv')
    # test_in, test_out = read_regression('data_regression/data.activation.test.100.csv')
    # train_in, train_out, test_in, test_out = normalize_regression(train_in, train_out, test_in, test_out)
    #
    # peek(train_in, train_out, test_in, test_out)
    #
    # ticks = {x: str(x) for x in [x for x in range(200, 10000, 200)]}
    # ticks[1] = str(1)
    #
    # app.layout = html.Div([
    #     # html.H4('Big text'),
    #     html.Center("Select iteration:"),
    #     dcc.Slider(id='slider', min=1, max=ITER, step=1, value=1, marks=ticks),
    #     dcc.Graph(id="graph", style={'width': '98vw', 'height': '90vh'}),
    # ])
    #
    # app.run_server(debug=True, use_reloader=False)
