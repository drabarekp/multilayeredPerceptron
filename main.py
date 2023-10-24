import numpy as np

from dash import Dash, dcc, html, Input, Output
from MlpBase import MlpBase
from figure import get_figure
from read_file import read_classification, read_regression

np.seterr(all='raise')
ITER = 100
LAYERS = [1, 4, 4, 1]
SEED = 1002

app = Dash('Network plot')
DATA = []   # data must be accessible for automatic calls


@app.callback(Output("graph", "figure"),
              Input('slider', 'value'))
def select_iteration(number):
    return get_figure(DATA[number - 1], LAYERS)


def peek(data_input, data_output, test_input, test_output):
    mlp = MlpBase(LAYERS, _seed=SEED)

    iterations = []
    for i in range(ITER):
        iteration_data = mlp.learn_iteration(data_input, data_output, test_input, test_output)
        iterations.append(iteration_data)

        if i % 10 == 9:
            print("ITERATION {:5d}: TRAIN ERROR = {:8.3f}, TEST ERROR = {:8.3f}".format(
                i + 1, iteration_data[4], iteration_data[5]))

    global DATA
    DATA = iterations

# def test_np():
#     val = np.array([1, 2, 3]) * np.transpose(np.array([5, 4, 3]))
#     print(val)


# def test_fn():
#     a = np.array([1, 2, 3])
#     b = np.array([4, 5, 7])
#
#     print(fn.sigmoid(a[1]))
#     print(fn.relu(a[1]))
#     print(fn.arctan(a[1]))
#
#     print(fn.sigmoid_derivative(a[1]))
#     print(fn.relu_derivative(a[1]))
#     print(fn.arctan_derivative(a[1]))
#
#     print(fn.sigmoid(a))
#     print(fn.relu(a))
#     print(fn.arctan(a))
#     print(fn.mean_squared_error(a, b))
#     print(fn.mean_absolute_error(a, b))
#
#     print(fn.sigmoid_derivative(a))
#     print(fn.relu_derivative(a))
#     print(fn.arctan_derivative(a))
#     print(fn.mean_squared_error_derivative(a, b))
#     print(fn.mean_absolute_error_derivative(a, b))


# def test_plots():
#     dc_in, dc_out = read_classification('data_classification/data.three_gauss.train.100.csv')
#     train_in, train_out = read_regression('data_regression/data.activation.train.100.csv')
#
#     mc = MlpBase([2, 8, 8, 3], _seed=1002)
#     mr = MlpBase([1, 4, 4, 1], _seed=1002)
#
#     plot_classification_points(dc_in, dc_out)
#     plot_classification_score(dc_in, dc_out, mc)
#     plot_classification_score_by_class(dc_in, dc_out, mc, 0)
#     plot_classification_score_by_class(dc_in, dc_out, mc, 1)
#     plot_classification_score_by_class(dc_in, dc_out, mc, 2)
#
#     plot_regression_points(train_in, train_out)
#     plot_regression_line(train_in, train_out, mr)


if __name__ == '__main__':
    # read_classification('data_classification/data.three_gauss.train.100.csv')

    train_in, train_out = read_regression('data_regression/data.activation.train.100.csv')
    test_in, test_out = read_regression('data_regression/data.activation.train.100.csv')
    peek(train_in, train_out, test_in, test_out)

    ticks = {x: str(x) for x in [x for x in range(200, 10000, 200)]}
    ticks[1] = str(1)

    app.layout = html.Div([
        # html.H4('Big text'),
        html.Center("Select iteration:"),
        dcc.Slider(id='slider', min=1, max=ITER, step=1, value=1, marks=ticks),
        dcc.Graph(id="graph", style={'width': '98vw', 'height': '90vh'}),
    ])

    app.run_server(debug=True, use_reloader=False)
