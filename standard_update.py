import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import argparse
import math

parser = argparse.ArgumentParser()
parser.add_argument('--path', action="append", help="Aggregation path")
parser.add_argument('--port', type=int, help="Port to start dash board on ", default=8050)
parser.add_argument('--metric', type=str, help="Port to start dash board on ", default='standard_evaluate')
parser.add_argument('--num_data', type=int, help="Number of runs used to average", default=10)
args = parser.parse_args()

import pandas as pd

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

colors = {
    'background': '#111111',
    'text': '#7FDBFF'
}

layout = {
            'plot_bgcolor': colors['background'],
            'paper_bgcolor': colors['background'],
            'font': {
                'color': colors['text']
            }
        }
dfs = {}
tasks = ['trec', 'sst', 'cola', 'subjectivity']

class LoadDatasets:
    def __init__(self, args, tasks=['trec', 'sst', 'cola', 'subjectivity']):
        self.args = args
        # Standard Evaluate path
        self.se_path = [] 
        self.fm_path = [] 
        self.dg_path = [] 
        # Forgetting Metric Path
        if args.path:
            for pth in args.path:
                self.se_path.append(pth + '/evaluate_csv_agg/aggregates/'+ args.metric +'/')
                self.fm_path.append(pth + 'evaluate_csv_agg/aggregates/forgetting_metric/standard_total.df')
                self.dg_path.append(pth + 'evaluate_csv_agg/aggregates/task_diagnostics/overall.df')
        self.df = {}
        self.tasks = tasks
        self.load_tasks()

    def load_tasks(self):
        # Load Tasks ds
        self.ltotal = []
        self.ltask_diag = []
        self.ldf = {}
        for task in self.tasks:
            self.ldf[task] = []
        for fm_pth, dg_pth, se_pth in zip(self.fm_path, self.dg_path, self.se_path):
            self.ltotal.append(pd.read_pickle(fm_pth))
            self.ltask_diag.append(pd.read_pickle(dg_pth))
            for task in self.tasks:
                self.ldf[task].append(pd.read_pickle(se_pth + task + '.df'))

        self.total = pd.concat(self.ltotal, ignore_index=True)
        self.task_diag = pd.concat(self.ltask_diag, ignore_index=True)
        for task in self.tasks:
            self.df[task] = pd.concat(self.ldf[task], ignore_index=True)

    def get_unique(self, attr='code'):
        output = []
        for task in self.tasks:
            output.extend(list(self.df[task][attr].unique()))
        output = list(set(output))
        if attr == 'code':
            output.append('all')
        return output

dataset = LoadDatasets(args)

app.layout = html.Div(style={'backgroundColor': colors['background']},
    children=[
       html.H1(
            children='Forgetting Analyser',
            style={
                'textAlign': 'center',
                'color': colors['text']
            }
        ),
        html.Div(children='Forgetting Analyser', style={
            'textAlign': 'center',
            'color': colors['text']
        }),
        dcc.Dropdown(
            id='code',
            options=[{'label': i, 'value': i} for i in dataset.get_unique()],
            value='all'
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
            value=['100'],
            multi=True
        ),
        dcc.Dropdown(
            id='layer',
            options=[{'label': i, 'value': i} for i in dataset.get_unique('layer')],
            value=['1'],
            multi=True),
        dcc.Dropdown(
            id='tasks',
            options=[{'label': i, 'value': i} for i in dataset.tasks],
            value=dataset.tasks,
            multi=True),
        dcc.Tabs([
        dcc.Tab(label='Performance', children=[dcc.Graph(id='performance_plot')]),
        dcc.Tab(label='Forgetting Metric', children=[
        html.Div(children='Forgetting Metric', style={
            'textAlign': 'center',
            'color': colors['text']
        }),
        dcc.Graph(id='forgetting_plot'),
        html.Div(children='Task Diagnostic classifier', style={
            'textAlign': 'center',
            'color': colors['text']
        }),
        dcc.Graph(id='diagnostic_plot')]),
        dcc.Tab(label='Average Accuracy', children=[ 
        dcc.Graph(id='forgetting_plot_copy'),
        html.Div(children='Task Diagnostic classifier', style={
            'textAlign': 'center',
            'color': colors['text']
        }),
        dcc.Graph(id='diagnostic_plot_copy')]),
        ]),
])

def get_name(current_row, task, exper, hdim, layer):
    name=''
    if len(exper) > 1:
        name += "E_"+ str(current_row['exper'])
    if len(hdim) > 1:
        name += "_H_"+ str(current_row['hdim'])
    if len(layer) > 1:
        name += "_L_" + str(current_row['layer'])
    if len(name) == 0 or len(tasks) > 1:
        name += "_T_" + task
    return name

def get_prefix(total):
    common_prefix=''
    index = 0
    while True:
        char=None
        for i in total.exper.unique():
            if index >= len(i):
                return ''
            curr_char = i[index]
            if not char:
                char = curr_char
            else:
                if char != curr_char:
                    return common_prefix
        common_prefix += curr_char
        index += 1

def filter_df(this_df, code, exper, hdim, layer, tasks):
    this_df = this_df[this_df['code'] == code] if not code == 'all' else this_df
    this_df = this_df[this_df.exper.isin(exper)]
    this_df = this_df[this_df.hdim.isin(hdim)]
    this_df = this_df[this_df.layer.isin(layer)]
    prefix = ''
    if len(this_df) > 1:
        prefix = get_prefix(this_df)
    this_df.exper = this_df.exper.str.replace(prefix, '')
    this_df.exper = this_df.exper.str.replace('_exper', '')
    if code == 'all':
        this_df = this_df.groupby(['hdim', 'exper', 'layer']).mean().reset_index()
    return this_df

def get_bar_data(bar_df, step):
    bar_data = []
    # Confidence interval multiplier
    ci_mult = 1.960/math.sqrt(args.num_data)
    for i in range(len(bar_df)):
        current_row = bar_df.iloc[i]
        name = "L_" + str(current_row['layer']) + "_H_" + str(current_row['hdim'])
        bar_data.append({'type': 'bar',
              'x': [name],
              'y': [current_row[step + '_mean']],
              'error_y': dict(type='data', array=[ci_mult*current_row[step+'_std']]),
              'name': current_row['exper'].upper()
              })
    return {'data':  bar_data, 'layout': layout}



@app.callback(
    [Output('performance_plot', 'figure'),
     Output('forgetting_plot', 'figure'),
     Output('diagnostic_plot', 'figure'),
     Output('forgetting_plot_copy', 'figure'),
     Output('diagnostic_plot_copy', 'figure')],
    [Input('code', 'value'),
     Input('exper', 'value'),
     Input('hdim', 'value'),
     Input('layer', 'value'),
     Input('tasks', 'value')])
def update_graph(code, exper, hdim, layer, tasks):
    data = []
    splitcode = code.split('_')
    for task in tasks:
        my_df = filter_df(dataset.df[task], code, exper, hdim, layer, tasks)
        if len(my_df) == 0:
            print("Not found this config", exper, hdim, layer, code)
            continue
        for i in range(len(my_df)):
            current_row = my_df.iloc[i]
            accuracy = [current_row['step_1_mean'], current_row['step_2_mean'], current_row['step_3_mean'], current_row['step_4_mean']]
            variance = [current_row['step_1_var'], current_row['step_2_var'], current_row['step_3_var'], current_row['step_4_var']]
            rr = {'exper': current_row['exper'],
                  'type': 'line', 'x': splitcode, 
                  'y': accuracy,
                  'error_y': dict(type='data', array=variance),
                  'name': get_name(current_row, task, exper, hdim, layer) }
            data.append(rr)
    pp = {'data': data, 'layout': layout}

    fp = get_bar_data(filter_df(dataset.total, code, exper, hdim, layer, tasks), 'step_2')
    dp = get_bar_data(filter_df(dataset.task_diag, code, exper, hdim, layer, tasks), 'step_0')
    return pp,fp,dp,fp,dp


if __name__ == '__main__':
    app.run_server(debug=True, port=args.port)
