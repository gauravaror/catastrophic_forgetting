import pandas as pd
import matplotlib.pyplot as plta
import matplotlib.text as txt
from matplotlib.lines import Line2D
import numpy as np
import scipy.stats
import sys

import argparse

parser = argparse.ArgumentParser(description='Plots the graphs using current_dataset.df on our expeirmentss.')
parser.add_argument('--show_config', default=1, type=int, help='Default task to show results for : 1: _sst_cola_subjectivity_trec 2: _cola_trec_sst_subjectivity  3: _trec_sst_subjectivity_cola')
parser.add_argument('--show_exper', default=1, type=int, help='Default experiment to show results for : 1: lstm 2: cnn')
parser.add_argument('--show_hd',action="store_true", help='Show the h_dimension as lines')

parser.add_argument('--h_dim', action='append')

parser.add_argument('--data', action='append',
                              help="dead_per, accuracy, avg_zeros_per, weight_corr, corr")

parser.add_argument('--inte_lay', action='append',
                                  type=int,
                                  help="Internal layers to show 0,1,2")
parser.add_argument('--gram', 
                     action='append',
                     default=[0],
                     help="Which gram to show 0,1,2 values depends on if data has it")

parser.add_argument('--layer', action='append', type=int)
parser.add_argument('--task', action='append')
args = parser.parse_args()


if not args.data:
  print("Not asked to plot anything: ploting weight_corr")
  args.data = ['dead_per','avg_zeros_per','weight_corr','corr']


if not args.inte_lay:
  args.inte_lay = []
  max_=max(args.layer)
  for i in range(max_):
    args.inte_lay.append(i)

def mean_confidence_interval(data, confidence=0.95):
    a = 1.0 * np.array(data)
    n = len(a)
    m, se = np.mean(a), scipy.stats.sem(a)
    h = se * scipy.stats.t.ppf((1 + confidence) / 2., n-1)
    return m, m-h, m+h



# COdes available "trec_sst_subject_cola" "trec_sst_subject_cola" "sst_cola_subjectivity_trec" "subjectivity_trec_cola_sst" "sst_trec_subjectivity_cola"
code_mapping = {
'sst_trec_subjectivity_cola': { 'first': 'sst', 'second': 'trec', 'third': 'subjectivity', 'fourth': 'cola'},
'trec_sst_subject_cola': { 'first': 'trec', 'second': 'sst', 'third': 'subjectivity', 'fourth': 'cola'},
'sst_cola_subjectivity_trec': { 'first': 'sst', 'second': 'cola', 'third': 'subjectivity', 'fourth': 'trec'},
'subjectivity_trec_cola_sst': { 'first': 'subjectivity', 'second': 'trec', 'third': 'cola', 'fourth': 'sst'},
}

majority = {'subjectivity': 0.5, 'sst': 0.2534059946, 'trec': 0.188, 'cola': 0.692599620493358}

sota = {'subjectivity': 0.955, 'sst': 0.547, 'trec': 0.9807, 'cola': 0.772}
tasks={'sst': 'b', 'cola': 'g','subjectivity': 'r', 'trec': 'c'}

custom_lines=[]
legend_labels=[]

for task in tasks.keys():
  custom_lines.append(Line2D([0], [0], color=tasks[task], lw=1))
  legend_labels.append(task.upper())
  
if args.show_hd:
  h_dim = {100:'-' , 400:'-',  900:'--', 1400:'--', 1900:'-.', 2400:':'}
  h_dim_marker = {100:'o' , 400:'o',  900:'D', 1400:'D', 1900:'d', 2400:'d'}
  h_dim_fco = {100:'y' , 400:'b',  900:'y', 1400:'b', 1900:'y', 2400:'b'}

  for dim in h_dim.keys():
    custom_lines.append(Line2D([0], [0], markerfacecolor=h_dim_fco[dim], marker=h_dim_marker[dim], lw=1))
    legend_labels.append("Dimension: " + str(dim))

  num_layers={1: '-', 2: '--', 3: '-.', 4: ':', 5: ':'}

  for layer in num_layers.keys():
    custom_lines.append(Line2D([0], [0], linestyle=num_layers[layer], lw=1))
    legend_labels.append("Layer: " + str(layer))

else:
  h_dim = {100:'.' , 200: ',', 300:'o', 400:'v', 500:'^', 600:'<', 700:'>', 800:'1', 900:'2', 1000:'3', 1100:'4', 1200:'s', 1300:'p', 1400:'*', 1500:'h', 1600:'H', 1700:'+', 1800:'x', 1900:'D', 2000:'d',2100:'|', 2400:'_'}

#  for dim in h_dim.keys():
#    custom_lines.append(Line2D([0], [0], marker=h_dim[dim], lw=1))
#    legend_labels.append("Dimension: " + str(dim))

  num_layers={1: '-', 2: '--', 3: '-.', 4: ':'}

  for layer in num_layers.keys():
    custom_lines.append(Line2D([0], [0], linestyle=num_layers[layer], lw=1))
    legend_labels.append("Layer: " + str(layer))

if args.h_dim:
  show_dim=args.h_dim
else:
  show_dim=[100,400,900,1400,1900]

if args.layer:
  show_layer=args.layer
else:
  show_layer=[1,3,2,4]

show_experiment="cnn"
if args.task:
  show_tasks=args.task
else:
  show_tasks=[ 'subjectivity', 'sst', 'trec']

if args.show_config == 1:
  show_config='_sst_cola_subjectivity_trec'
elif args.show_config == 2:
  show_config='_cola_trec_sst_subjectivity'
elif args.show_config == 3:
  show_config='_trec_sst_subjectivity_cola'
else:
  print("Wrong config selected, selecting _trec_sst_subjectivity_cola")
  show_config='_trec_sst_subjectivity_cola'

if args.show_exper == 1:
  show_experiment="lstm"
elif args.show_exper == 2:
  show_experiment="cnn"
elif args.show_exper == 3:
  show_experiment="gru"
elif args.show_exper == 4:
  show_experiment="cnn_stacked"
else:
  print("Wrong Experiment selected, selecting lstm")
  show_experiment="lstm"

if len(show_layer) == 1 and len(args.inte_lay)>1:
  print("Showing inte layer")
  diff_inte_lay=True
else: 
  diff_inte_lay=False

task_order=show_config.split("_")[1:]


def configure_subplot(df,plt,d):
  for l_id,layer_ in enumerate(show_layer):
    for h_did,h_d_ in enumerate(show_dim):
      for task in show_tasks:
        for g in args.gram:
          for la in args.inte_lay:
            layer=int(layer_)
            h_d=int(h_d_)
            plot_list=[]
            ord=0
            for t in task_order:
              sent_data=df[df.layer==layer][df.h_dim==h_d][df.task==t][df.evaluate==task][df.lay==la][df.gram==g]
              #print("Plotting data", sent_data)
              # Get the data value to display
              if len(sent_data) >0:
                plot_value=sent_data.iloc[0][d]
              else:
                plot_value=None
                print("#NOTAVAILABLE  Current layer %s, h_dim %s, task %s , evaluate %s, lay %s, gram %s"%( 
                        str(layer),str(h_d),t,task,str(la),
                        str(g)),file=sys.stderr)
              plot_list.append(plot_value)
              #print("Current layer %s, h_dim %s, task %s , evaluate %s, lay %s, gram %s plot_value %s showing_data %s sent_data_len %s"%( 
                        #str(layer),str(h_d),t,task,str(la),
                        #str(g),str(plot_value),str(d),str(len(sent_data))))
            if args.show_hd:
              if diff_inte_lay:
                 sho_layer=(la+1)
              else:
                 sho_layer=layer
              #print("Ploting list", plot_list)
              plt.plot(plot_list, markerfacecolor=h_dim_fco[h_d],
                       marker=h_dim_marker[h_d], color=tasks[task],
                       linestyle=num_layers[sho_layer])
            else:
              #print("Ploting list", plot_list)
              plt.plot(plot_list,markerfacecolor='b')
  my_xticks = task_order
  my_xticks_color = [tasks[t] for t in task_order]
  plta.xticks(range(4), my_xticks)
  [t.set_color(i) for (i,t) in
    zip( my_xticks_color, plt.xaxis.get_ticklabels())]
  plt.set_title(d.upper())

fig=plta.figure()
df=pd.read_pickle('final_corrv2.df')
for i,met in enumerate(args.data):
  plt = plta.subplot(len(args.data)/2,len(args.data) - (len(args.data)/2),i+1)
  configure_subplot(df,plt,met)

plta.legend(custom_lines,legend_labels)
fig.suptitle("Layer"+str(args.layer[0]).upper())
plta.show()
