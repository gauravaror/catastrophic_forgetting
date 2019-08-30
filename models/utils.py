from collections import Counter
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import tensorflow as tf
import io
import PIL.Image
from torchvision.transforms import ToTensor
import torch
import numpy as np
from models.trec import TrecDatasetReader
from models.subjectivity import SubjectivityDatasetReader
from models.CoLA import CoLADatasetReader
from models.ag import AGNewsDatasetReader
from allennlp.data.dataset_readers import DatasetReader
from models.sst import StanfordSentimentTreeBankDatasetReader1


def load_dataset(code, train_data, dev_data, few_data):
  if code == "sst_2c":
    # Sentiment task 2 class
    reader_senti_2class = StanfordSentimentTreeBankDatasetReader1(granularity="2-class")
    train_data["sst_2c"] = reader_senti_2class.read('data/SST/trees/train.txt')
    dev_data["sst_2c"] = reader_senti_2class.read('data/SST/trees/dev.txt')
    few_data["sst_2c"] = reader_senti_2class.read('data/SST/trees/few.txt')
  elif code == 'sst':
    reader_senti = StanfordSentimentTreeBankDatasetReader1()
    train_data["sst"] = reader_senti.read('data/SST/trees/train.txt')
    dev_data["sst"] = reader_senti.read('data/SST/trees/dev.txt')
    few_data["sst"] = reader_senti.read('data/SST/trees/few.txt')
  elif code == 'cola':
    reader_cola = CoLADatasetReader()
    train_data["cola"] = reader_cola.read('data/CoLA/train.txt')
    dev_data["cola"] = reader_cola.read('data/CoLA/dev.txt')
    few_data["cola"] = reader_cola.read('data/CoLA/few.txt')
  elif code == 'trec':
    reader_trec = TrecDatasetReader()
    train_data["trec"] = reader_trec.read('data/TREC/train.txt')
    dev_data["trec"] = reader_trec.read('data/TREC/dev.txt')
    few_data["trec"] = reader_trec.read('data/TREC/few.txt')
  elif code == 'subjectivity':
    reader_subj = SubjectivityDatasetReader()
    train_data["subjectivity"] = reader_subj.read('data/Subjectivity/train.txt')
    dev_data["subjectivity"] = reader_subj.read('data/Subjectivity/test.txt')
    few_data["subjectivity"] = reader_subj.read('data/Subjectivity/few.txt')
  elif code == 'ag':
    reader_ag = AGNewsDatasetReader()
    train_data["ag"] = reader_ag.read('data/ag/train.csv')
    dev_data["ag"] = reader_ag.read('data/ag/val.csv')
    few_data["ag"] = reader_ag.read('data/ag/val.csv')
  else:
    print("Unknown Task code provided")

def gen_plot(plt):
    """Create a pyplot plot and save to buffer."""
    buf = io.BytesIO()
    plt.savefig(buf, format='jpeg')
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = ToTensor()(image).unsqueeze(0)
    return image

def run_tsne_embeddings(data_view_tsne, labels_orig, train, evaluate, getlayer, gram, labels_map, mean = None):
  plt.clf()
  tsne_model = TSNE(n_components=2, perplexity=30.0)
  tnse_embedding = tsne_model.fit_transform(data_view_tsne)
  index_color = {0: 'b.', 1: 'g.', 2: 'r.', 3: 'c.', 4: 'm.', 5: 'y.'}
  legend_tracker = {0: 'b.', 1: 'g.', 2: 'r.', 3: 'c.', 4: 'm.', 5: 'y.'}
  mean_color = {0: 'b^', 1: 'g^', 2: 'r^', 3: 'c^', 4: 'm^', 5: 'y^'}

  if mean:
    mean_keys = list(mean.keys())
    mean_values = list(mean.values())

    mean_val_tp = torch.stack(mean_values).cpu().numpy()

    combined_tsne = np.append(data_view_tsne, mean_val_tp, axis=0)
    tnse_embedding = tsne_model.fit_transform(combined_tsne)

    # Boundaries of each of mean and actual data points.
    starting_labels = len(data_view_tsne)
    starting_training_encoder = starting_labels + len(mean_keys)
    fig, axes = plt.subplots(1+len(mean_keys),1)
    for i in range(starting_labels, starting_training_encoder):
      axes[len(mean_keys)].plot(tnse_embedding[i][0], tnse_embedding[i][1], mean_color[mean_keys[i - len(data_view_tsne)]])
  else:
    print("Printing labels for ", len(set(labels_orig)))
    fig, axes = plt.subplots(len(set(labels_orig)),1, sharex='row')

  task_label = labels_map[evaluate]
  for i in range(0, len(data_view_tsne)):
    if labels_orig[i] in legend_tracker:
      axes[labels_orig[i]].plot(tnse_embedding[i][0], tnse_embedding[i][1], index_color[labels_orig[i]], label=task_label[labels_orig[i]])
      axes[labels_orig[i]].legend(loc='upper right')
      legend_tracker.pop(labels_orig[i])
    else:
      axes[labels_orig[i]].plot(tnse_embedding[i][0], tnse_embedding[i][1], index_color[labels_orig[i]])
  plt.legend()
  image_plot = gen_plot(plt)
  plt.close('all')
  return image_plot

def get_catastrophic_metric(tasks, metrics):
     forgetting_metrics = Counter()
     count_task = Counter()
     forgetting={'total': 0}
     for i,task in enumerate(tasks):
         for j in range(i):
             print("Calculating backward for ",tasks[i] ,
                   " taking calculation at tasks", tasks[j])
             step = i-j
             current_forgetting = (metrics[tasks[j]][tasks[j]] - metrics[tasks[i]][tasks[j]])
             if step > 0:
               forgetting_metrics[str(step) + "_step"] += current_forgetting
               count_task[str(step) + "_step"] += 1
             forgetting_metrics[tasks[j]] += current_forgetting
             count_task[tasks[j]] += 1
     for metric in forgetting_metrics:
         if metric in tasks:
           print("Calculating forgetting for", forgetting_metrics[task], count_task[task])
           forgetting[metric] = forgetting_metrics[metric] / count_task[metric]
           forgetting['total'] += forgetting[metric]
         else:
           forgetting[metric] = forgetting_metrics[metric] / count_task[metric]
     
     # Calculate total forgetting of all the
     length_tasks = len(tasks) - 1
     if length_tasks > 1:
         forgetting['total'] = (forgetting['total']/(len(tasks) - 1))

     return forgetting
