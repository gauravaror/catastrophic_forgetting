import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('path', type=str, help="Aggregation path")
args = parser.parse_args()

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
    def __init__(self, args, tasks=['trec', 'sst', 'cola', 'subjectivity']):
        self.args = args
        # Standard Evaluate path
        self.se_path = 'standard_evaluate/'
        # Forgetting Metric Path
        self.fm_path = 'forgetting_metic/standard_total.df'
        if args.path:
            self.se_path = args.path + '/evaluate_csv_agg/aggregates/standard_evaluate/'
            self.fm_path = args.path + 'evaluate_csv_agg/aggregates/forgetting_metric/standard_total.df'
        self.df = {}
        self.tasks = tasks
        self.load_tasks()

    def load_tasks(self):
        # Load Tasks ds
        self.total = pd.read_pickle(self.fm_path)
        for task in self.tasks:
            self.df[task] = pd.read_pickle(self.se_path + task + '.df')

    def get_unique(self, attr='code'):
        output = []
        for task in self.tasks:
            output.extend(list(self.df[task][attr].unique()))
        return list(set(output))

dataset = LoadDatasets(args)

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
        dcc.Graph(id='forgetting_plot'),
])

@app.callback(
    [Output('performance_plot', 'figure'),
     Output('forgetting_plot', 'figure')],
    [Input('code', 'value'),
     Input('exper', 'value'),
     Input('hdim', 'value'),
     Input('layer', 'value'),
     Input('tasks', 'value')])
def update_graph(code, exper, hdim, layer, tasks):
    df = dataset.df
    total = dataset.total
    data = []
    splitcode = code.split('_')
    def get_name(current_row, task):
        if len(exper) > 1:
            return "Exper: "+ str(current_row['exper'])
        elif len(tasks) > 1:
            return "Task: "+ task
        elif len(hdim) > 1:
            return "HDIM: "+ str(current_row['hdim'])
        elif len(layer) > 1:
            return "Layer: " + str(current_row['layer'])
        return "Task: " + task

    def filter_df(this_df):
        this_df = this_df[this_df['code'] == code]
        this_df = this_df[this_df.exper.isin(exper)]
        this_df = this_df[this_df.hdim.isin(hdim)]
        this_df = this_df[this_df.layer.isin(layer)]
        return this_df

    for task in tasks:
        my_df = filter_df(df[task])
        if len(my_df) == 0:
            print("Not found this config", exper, hdim, layer, code)
            continue
        for i in range(len(my_df)):
            current_row = my_df.iloc[i]
            accuracy = [current_row['step_1_mean'], current_row['step_2_mean'], current_row['step_3_mean'], current_row['step_4_mean']]
            rr = {'exper': current_row['exper'], 'type': 'line', 'x': splitcode, 'y': accuracy, 'name': get_name(current_row, task) }
            print(rr)
            data.append(rr)

    pp = {
        'data': data,
        'layout': {
                'plot_bgcolor': colors['background'],
                'paper_bgcolor': colors['background'],
                'font': {
                    'color': colors['text']
                }
            }
        }
    fp = {

            }
    return pp,pp


if __name__ == '__main__':
    app.run_server(debug=True)
