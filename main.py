from dash import Dash, dcc, html, Input, Output
from figure import get_figure
from functions import *
from MlpBase import MlpBase
from plots import plot_classification, plot_regression
from read_file import read_classification, read_regression, normalize_regression

np.seterr(all='raise')
DESC_LENGTH = 0.003
ITER = 10000
LAYERS = [2, 4, 6, 4, 3]
SEED = 1002

app = Dash('Network plot')
DATA = []   # data must be accessible for automatic calls


@app.callback(Output('graph', 'figure'),
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
                  descent_length=DESC_LENGTH)

    iterations = []
    for i in range(ITER):
        iteration_data = mlp.learn_iteration(data_input, data_output, test_input, test_output)
        iterations.append(iteration_data)

        if i % 10 == 9:
            print('ITERATION {:6d}: TRAIN ERROR = {:7.4f}, TEST ERROR = {:7.4f}'.format(
                i + 1, iteration_data[4], iteration_data[5]))

    global DATA
    DATA = iterations

    return mlp


def show_network():
    ticks = {x: str(x) for x in [x for x in range(200, 10000, 200)]}
    if ITER not in ticks.keys():
        ticks[ITER] = str(ITER)
    ticks[1] = str(1)

    app.layout = html.Div([
        dcc.Slider(id='slider', min=1, max=ITER, step=1, value=1, marks=ticks),
        dcc.Graph(id='graph', style={'width': '98vw', 'height': '95vh'}),
    ])

    app.run_server(debug=True, use_reloader=False)


if __name__ == '__main__':
    train_in, train_out = read_classification('data_classification/data.three_gauss.train.100.csv')
    test_in, test_out = read_classification('data_classification/data.three_gauss.test.100.csv')

    # train_in, train_out = read_regression('data_regression/data.activation.train.100.csv')
    # test_in, test_out = read_regression('data_regression/data.activation.test.100.csv')
    # train_in, train_out, test_in, test_out = normalize_regression(train_in, train_out, test_in, test_out)

    model = peek(train_in, train_out, test_in, test_out)

    plot_classification(train_in, train_out, test_in, test_out, model)
    # plot_regression(train_in, train_out, test_in, test_out, model)

    # show_network()
