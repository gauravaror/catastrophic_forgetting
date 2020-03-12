import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('path', type=str, help="Aggregation path")
parser.add_argument('--port', type=int, help="Port to start dash board on ", default=8050)
parser.add_argument('--metric', type=str, help="Port to start dash board on ", default='standard_evaluate')
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
        self.se_path = args.metric + '/'
        # Forgetting Metric Path
        self.fm_path = 'forgetting_metic/standard_total.df'
        if args.path:
            self.se_path = args.path + '/evaluate_csv_agg/aggregates/'+ args.metric +'/'
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

    def filter_df(this_df):
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


    for task in tasks:
        my_df = filter_df(df[task])
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
                  'name': get_name(current_row, task) }
            print(rr)
            data.append(rr)
    layout = {
                'plot_bgcolor': colors['background'],
                'paper_bgcolor': colors['background'],
                'font': {
                    'color': colors['text']
                }
            }
    pp = {'data': data, 'layout': layout}

    total_data =  []
    tot_df = filter_df(total)
    for i in range(len(tot_df)):
        current_row = tot_df.iloc[i]
        name = "L_" + str(current_row['layer']) + "_H_" + str(current_row['hdim'])
        rr = {'type': 'bar', 'x': [current_row['exper']], 'y': [current_row['step_2_mean']], 'error_y': dict(type='data', array=[current_row['step_2_var']]),'name': name}
        total_data.append(rr)
    print(total_data)
    fp = {'data':  total_data, 'layout': layout}
    return pp,fp


if __name__ == '__main__':
    app.run_server(debug=True, port=args.port)
