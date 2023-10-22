# import plotly.express as px
from dash import Dash, dcc, html, Input, Output
from figure import get_figure

app = Dash('Network plot')
DATA = None     # data must be accessible for automatic calls
LAYERS = None


@app.callback(Output("graph", "figure"),
              Input('slider', 'value'))
def select_iteration(number):
    # df = px.data.tips()  # replace with your own data source
    # fig = px.scatter(df, x="total_bill", y="tip",
    #                  facet_col="sex", height=400)
    # fig.update_layout(
    #     margin=dict(l=20, r=20, t=20, b=20),
    #     paper_bgcolor="LightSteelBlue", )
    # fig.update_layout(width=int(number))
    return get_figure(DATA, LAYERS)


def plot_iterations(data, layers):
    global DATA, LAYERS
    DATA = data
    LAYERS = layers

    dic = {x: str(x) for x in [x for x in range(500, 10000, 500)]}
    dic[1] = str(1)

    app.layout = html.Div([
        # html.H4('Live adjustable graph-size'),
        html.Center("Select iteration:"),
        dcc.Slider(id='slider', min=1, max=len(data), step=1, value=1, marks=dic),
        dcc.Graph(id="graph"),
    ])

    app.run_server(debug=True)
