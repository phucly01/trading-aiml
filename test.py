import dash
from dash.dependencies import Input, Output, State
from dash import dcc
from dash import html
import plotly.graph_objs as go

app = dash.Dash()

app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})

app.layout = html.Div([
	
	html.H3('Visualization'),
	html.Div([

		html.Label('Slider1'),
        dcc.Slider(
        id='slider1',
        min=0,
        max=50,
        value=10,
        step=1
        ),

        html.Label('Slider2'),
        dcc.Slider(
        id='slider2',
        min=0,
        max=50,
        value=10,
        step=1
        ),

        html.Div(id = 'slider1-show'),
        html.Div(id = 'slider2-show'),

        html.Hr(),

        dcc.Graph(id='test-figure', animate=True)

		])

	])

@app.callback(
    Output('slider1-show', 'children'),
    [Input('slider1', 'value')])

def deletion_show(slider1_val):
	return 'Slider1: ' + str(slider1_val)

@app.callback(
    Output('slider2-show', 'children'),
    [Input('slider2', 'value')])

def insertion_show(slider2_val):
	return 'Slider2: ' + str(slider2_val)


@app.callback(
    Output('test-figure', 'figure'),
    [Input('slider1', 'value'),
    Input('slider2', 'value')])

def visualize(slider1, slider2):

	data = [go.Bar(x=range(slider1), y=range(slider1)), go.Bar(x=range(slider2), y=range(slider2))]
	layout = go.Layout(title='test')

	figure = go.Figure(data = data, layout = layout)

	return figure

if __name__ == '__main__':
	app.run_server(debug = True)