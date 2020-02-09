import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

import pandas as pd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

dfs = {}
tasks = ['trec', 'sst', 'cola', 'subjectivity']
colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

class LoadDatasets:
    def __init__(self, path='', tasks=['trec', 'sst', 'cola', 'subjectivity']):
        self.path = path
        self.df = {}
        self.tasks = tasks
        self.load_tasks()

    def load_tasks(self):
        # Load Tasks ds
        for task in self.tasks:
            self.df[task] = pd.read_pickle(self.path + task + '.df')

    def get_unique(self, attr='code'):
        output = []
        for task in self.tasks:
            output.extend(list(self.df[task][attr].unique()))
        return list(set(output))

dataset = LoadDatasets()

app.layout = html.Div([
        dcc.Dropdown(
            id='code',
            options=[{'label': i, 'value': i} for i in dataset.get_unique()],
            value=dataset.get_unique()[0]
        ),
        dcc.Dropdown(
            id='exper',
            options=[{'label': i, 'value': i} for i in dataset.get_unique('exper')],
            value=dataset.get_unique('exper')[:1],
            multi=True
        ),
        dcc.Dropdown(
            id='hdim',
            options=[{'label': i, 'value': i} for i in dataset.get_unique('hdim')],
            value=dataset.get_unique('hdim')[:1],
            multi=True
        ),
        dcc.Dropdown(
            id='layer',
            options=[{'label': i, 'value': i} for i in dataset.get_unique('layer')],
            value=dataset.get_unique('layer')[:1],
            multi=True),
        dcc.Dropdown(
            id='tasks',
            options=[{'label': i, 'value': i} for i in dataset.tasks],
            value=dataset.tasks,
            multi=True),
        dcc.Graph(id='performance_plot'),
])

@app.callback(
    Output('performance_plot', 'figure'),
    [Input('code', 'value'),
     Input('exper', 'value'),
     Input('hdim', 'value'),
     Input('layer', 'value'),
     Input('tasks', 'value')])
def update_graph(code, exper, hdim, layer, tasks):
    dff = {}
    df = dataset.df
    data = []
    splitcode = code.split('_')
    def get_name(df, i, task):
        if len(exper) > 1:
            return "Exper: "+ str(df.iloc[i, 1])
        elif len(tasks) > 1:
            return "Task: "+ task
        elif len(hdim) > 1:
            return "HDIM: "+ str(df.iloc[i, 2])
        elif len(layer) > 1:
            return "Layer: " + str(df.iloc[i, 3])
        return "Task: " + task

    for task in tasks:
        dff[task] = df[task][df[task]['code'] == code]
        dff[task] = dff[task][dff[task].exper.isin(exper)]
        dff[task] = dff[task][dff[task].hdim.isin(hdim)]
        dff[task] = dff[task][dff[task].layer.isin(layer)]
        for i in range(len(dff[task])):
            rr = {'exper':dff[task].iloc[i,1], 'type': 'line', 'x': splitcode, 'y':list(dff[task].iloc[i,7::6]), 'name': get_name(dff[task], i, task) }
            print(rr)
            data.append(rr)

    return {
        'data': data,
        'layout': {
                'plot_bgcolor': colors['background'],
                'paper_bgcolor': colors['background'],
                'font': {
                    'color': colors['text']
                }
            }
        }


if __name__ == '__main__':
    app.run_server(debug=True)
