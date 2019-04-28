import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


# COdes available "trec_sst_subject_cola" "trec_sst_subject_cola" "sst_cola_subjectivity_trec" "subjectivity_trec_cola_sst" "sst_trec_subjectivity_cola"
code_mapping = {
'sst_trec_subjectivity_cola': { 'first': 'sst', 'second': 'trec', 'third': 'subjectivity', 'fourth': 'cola'},
'trec_sst_subject_cola': { 'first': 'trec', 'second': 'sst', 'third': 'subjectivity', 'fourth': 'cola'},
'sst_cola_subjectivity_trec': { 'first': 'sst', 'second': 'cola', 'third': 'subjectivity', 'fourth': 'trec'},
'subjectivity_trec_cola_sst': { 'first': 'subjectivity', 'second': 'trec', 'third': 'cola', 'fourth': 'sst'},
}

majority = {'subjectivity': 0.5, 'sst': 0.2534059946, 'trec': 0.188, 'cola': 0.692599620493358}

sota = {'subjectivity': 0.955, 'sst': 0.547, 'trec': 0.9807, 'cola': 0.772}
h_dim = {100:'.' , 200: ',', 300:'o', 400:'v', 500:'^', 600:'<', 700:'>', 800:'1', 900:'2', 1000:'3', 1100:'4', 1200:'s', 1300:'p', 1400:'*', 1500:'h', 1600:'H', 1700:'+', 1800:'x', 1900:'D', 2000:'d',2100:'|', 2200:'_'}
num_layers={1: '-', 2: '--', 3: '-.', 4: ':'}
tasks={'sst': 'b', 'cola': 'g','subjectivity': 'r', 'trec': 'c'}

show_dim=[100,500,1000,1500,2000,2400]
show_dim=list(range(100,2500,100))
show_layer=[1,2,3]
show_tasks=[ 'subjectivity']
show_config='subjectivity_trec_cola_sst'

custom_lines = [Line2D([0], [0], color='b', lw=1),
                Line2D([0], [0], color='g', lw=1),
                Line2D([0], [0], color='r', lw=1),
		Line2D([0], [0], color='c', lw=1),
                Line2D([0], [0], linestyle='-', lw=1),
                Line2D([0], [0], linestyle='--', lw=1),
                Line2D([0], [0], linestyle='-.', lw=1),
                Line2D([0], [0], linestyle=':', lw=1)]

custom_lines_all = [Line2D([0], [0], color='b', lw=1),
                Line2D([0], [0], color='g', lw=1),
                Line2D([0], [0], color='r', lw=1),
		Line2D([0], [0], color='c', lw=1),
                Line2D([0], [0], linestyle='-', lw=1),
                Line2D([0], [0], linestyle='--', lw=1),
                Line2D([0], [0], linestyle='-.', lw=1),
                Line2D([0], [0], linestyle=':', lw=1),
                Line2D([0], [0], marker='.', lw=1),
                Line2D([0], [0], marker=',', lw=1),
                Line2D([0], [0], marker='o', lw=1),
                Line2D([0], [0], marker='v', lw=1),
                Line2D([0], [0], marker='^', lw=1),
                Line2D([0], [0], marker='<', lw=1),
                Line2D([0], [0], marker='>', lw=1),
                Line2D([0], [0], marker='1', lw=1),
                Line2D([0], [0], marker='2', lw=1),
                Line2D([0], [0], marker='3', lw=1),
                Line2D([0], [0], marker='4', lw=1),
                Line2D([0], [0], marker='s', lw=1),
                Line2D([0], [0], marker='p', lw=1),
                Line2D([0], [0], marker='*', lw=1),
                Line2D([0], [0], marker='h', lw=1),
                Line2D([0], [0], marker='H', lw=1),
                Line2D([0], [0], marker='+', lw=1),
                Line2D([0], [0], marker='x', lw=1),
                Line2D([0], [0], marker='D', lw=1),
                Line2D([0], [0], marker='d', lw=1),
                Line2D([0], [0], marker='|', lw=1),
                Line2D([0], [0], marker='_', lw=1)]

legend_labels=['sst', 'cola', 'subjectivity', 'trec', 'Layer 1', 'Layer 2', 'Layer 3', 'Layer 4']

legend_labels_all=['sst', 'cola', 'subjectivity', 'trec', 'Layer 1', 'Layer 2', 'Layer 3', 'Layer 4',
"Dim 100",
"Dim 200",
"Dim 300",
"Dim 400",
"Dim 500",
"Dim 600",
"Dim 700",
"Dim 800",
"Dim 900",
"Dim 1000",
"Dim 1100",
"Dim 1200",
"Dim 1300",
"Dim 1400",
"Dim 1500",
"Dim 1600",
"Dim 1700",
"Dim 1800",
"Dim 1900",
"Dim 2000",
"Dim 2100",
"Dim 2200"]

df=pd.read_csv('data.csv')
for i,j in df[df.code==show_config][df.layer<5][df.h_dim<2300].iterrows():
  code=""
  code+=tasks[j['task']]
  code+=h_dim[j['h_dim']]
  code+=num_layers[j['layer']]
  if j['h_dim'] in show_dim and j['layer'] in show_layer and j['task'] in show_tasks:
    code_m = code_mapping[show_config]
    array_value=[j['first']/sota[j['task']] , j['second']/sota[j['task']], j['third']/sota[j['task']], j['fourth']/sota[j['task']]]
    this_l =plt.plot(array_value, code,label=str(j['task']) + str(j['layer']) + str(j['h_dim']))
    print(j['h_dim'],j['task'],j['first'], array_value)
for i in majority:
  plt.hlines(y=majority[i]/sota[i],xmin=0,xmax=3,colors=tasks[i],linestyles='dashdot', label="majority")
  print("majority", majority[i]/sota[i])
my_xticks = list(code_mapping[show_config].values())

plt.legend(custom_lines,legend_labels)

plt.xticks(range(4), my_xticks)
plt.show()
