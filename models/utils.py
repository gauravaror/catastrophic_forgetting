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
import torch.optim as optim
from allennlp.training.util import move_optimizer_to_cuda, evaluate
import numpy as np
from models.trec import TrecDatasetReader
from models.subjectivity import SubjectivityDatasetReader
from models.CoLA import CoLADatasetReader
from models.ag import AGNewsDatasetReader
from allennlp.data.dataset_readers import DatasetReader
from models.sst import StanfordSentimentTreeBankDatasetReader1
from allennlp.modules.token_embedders.elmo_token_embedder import ElmoTokenEmbedder
from allennlp.modules.seq2vec_encoders.cnn_encoder  import CnnEncoder
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertEmbedder
from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
import torch.nn as nn
from torch.autograd import Function

# Imported from https://github.com/Wizaron/binary-stochastic-neurons/blob/master/utils.py
class Hardsigmoid(nn.Module):

    def __init__(self):
        super(Hardsigmoid, self).__init__()
        self.act = nn.Hardtanh()

    def forward(self, x):
        return (self.act(x) + 1.0) / 2.0

class RoundFunctionST(Function):
    """Rounds a tensor whose values are in [0, 1] to a tensor with values in {0, 1}"""

    @staticmethod
    def forward(ctx, input):
        """Forward pass
        Parameters
        ==========
        :param input: input tensor
        Returns
        =======
        :return: a tensor which is round(input)"""

        # We can cache arbitrary Tensors for use in the backward pass using the
        # save_for_backward method.
        # ctx.save_for_backward(input)

        return torch.round(input)

    @staticmethod
    def backward(ctx, grad_output):
        """In the backward pass we receive a tensor containing the gradient of the
        loss with respect to the output, and we need to compute the gradient of the
        loss with respect to the input.
        Parameters
        ==========
        :param grad_output: tensor that stores the gradients of the loss wrt. output
        Returns
        =======
        :return: tensor that stores the gradients of the loss wrt. input"""

        # This is a pattern that is very convenient - at the top of backward
        # unpack saved_tensors and initialize all gradients w.r.t. inputs to
        # None. Thanks to the fact that additional trailing Nones are
        # ignored, the return statement is simple even when the function has
        # optional inputs.
        # input, weight, bias = ctx.saved_variables

        return grad_output

class BernoulliFunctionST(Function):

    @staticmethod
    def forward(ctx, input):

        return torch.bernoulli(input)

    @staticmethod
    def backward(ctx, grad_output):

        return grad_output

RoundST = RoundFunctionST.apply
BernoulliST = BernoulliFunctionST.apply

def get_optimizer(opt, parameters, lr_, wdecay):
    if opt == 'adam':
        if wdecay:
            print("Using Weight decay", wdecay)
            myopt = optim.Adam(parameters, lr=lr_, betas=(0.9, 0.999), eps=1e-08, weight_decay=wdecay)
        else:
            myopt = optim.Adam(parameters, lr=lr_, betas=(0.9, 0.999), eps=1e-08)
    else:
        myopt =  optim.Adam(parameters, lr=lr_, betas=(0.9, 0.999), eps=1e-08)
    if torch.cuda.is_available():
        move_optimizer_to_cuda(myopt)
    return myopt

def load_dataset(code, train_data, dev_data, few_data, embeddings='default'):
  if code == "sst_2c":
    # Sentiment task 2 class
    reader_senti_2class = StanfordSentimentTreeBankDatasetReader1(granularity="2-class", embeddings=embeddings)
    train_data["sst_2c"] = reader_senti_2class.read('data/SST/trees/train.txt')
    dev_data["sst_2c"] = reader_senti_2class.read('data/SST/trees/dev.txt')
    few_data["sst_2c"] = reader_senti_2class.read('data/SST/trees/few.txt')
  elif code == 'sst':
    reader_senti = StanfordSentimentTreeBankDatasetReader1(embeddings=embeddings)
    train_data["sst"] = reader_senti.read('data/SST/trees/train.txt')
    dev_data["sst"] = reader_senti.read('data/SST/trees/dev.txt')
    few_data["sst"] = reader_senti.read('data/SST/trees/few.txt')
  elif code == 'cola':
    reader_cola = CoLADatasetReader(embeddings=embeddings)
    train_data["cola"] = reader_cola.read('data/CoLA/train.txt')
    dev_data["cola"] = reader_cola.read('data/CoLA/dev.txt')
    few_data["cola"] = reader_cola.read('data/CoLA/few.txt')
  elif code == 'trec':
    reader_trec = TrecDatasetReader(embeddings=embeddings)
    train_data["trec"] = reader_trec.read('data/TREC/train.txt')
    dev_data["trec"] = reader_trec.read('data/TREC/dev.txt')
    few_data["trec"] = reader_trec.read('data/TREC/few.txt')
  elif code == 'subjectivity':
    reader_subj = SubjectivityDatasetReader(embeddings=embeddings)
    train_data["subjectivity"] = reader_subj.read('data/Subjectivity/train.txt')
    dev_data["subjectivity"] = reader_subj.read('data/Subjectivity/test.txt')
    few_data["subjectivity"] = reader_subj.read('data/Subjectivity/few.txt')
  elif code == 'ag':
    reader_ag = AGNewsDatasetReader(embeddings=embeddings)
    train_data["ag"] = reader_ag.read('data/ag/train.csv')
    dev_data["ag"] = reader_ag.read('data/ag/val.csv')
    few_data["ag"] = reader_ag.read('data/ag/val.csv')
  else:
    print("Unknown Task code provided")

def gen_plot(plt):
    """Create a pyplot plot and save to buffer."""
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = PIL.Image.open(buf)
    image = ToTensor()(image).unsqueeze(0)
    return image

def run_tsne_embeddings(data_view_tsne, labels_orig, labels_map = None, mean = None, task=True, mlabels=None):
  plt.clf()
  tsne_model = TSNE(n_components=2, perplexity=30.0)
  tnse_embedding = tsne_model.fit_transform(data_view_tsne)
  index_color = {0: 'b.', 1: 'g.', 2: 'r.', 3: 'c.', 4: 'm.', 5: 'y.'}
  plain_color = {0: 'b', 1: 'g', 2: 'r', 3: 'c', 4: 'm', 5: 'y'}
  legend_tracker = {0: 'b.', 1: 'g.', 2: 'r.', 3: 'c.', 4: 'm.', 5: 'y.'}
  mean_color = {0: 'b^', 1: 'g^', 2: 'r^', 3: 'c^', 4: 'm^', 5: 'y^'}
  markers = {0: ',', 1: '1', 2: 'x', 3: '+', 4: '|', 5: '_'}
  #markers = {0: '^', 1: '.', 2: 'X', 3: 'v', 4: '1', 5: '*'}

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
    plot_dim = len(set(labels_orig))
    if task:
        plot_dim = 1
    fig, axes = plt.subplots(plot_dim,1, sharex='row')

  task_label = None
  for i in range(0, len(data_view_tsne)):
    marker = index_color[labels_orig[i]]
    if not mlabels is None:
      marker = plain_color[labels_orig[i]]
      marker += markers[mlabels[i]]
    if labels_orig[i] in legend_tracker:
      label_ = labels_orig[i]
      if labels_map:
          label_ = labels_map[labels_orig[i]]
      axes.plot(tnse_embedding[i][0], tnse_embedding[i][1], marker, label=label_)
      axes.legend(loc='upper right')
      legend_tracker.pop(labels_orig[i])
    else:
      axes.plot(tnse_embedding[i][0], tnse_embedding[i][1], marker)
  plt.legend()
  image_plot = gen_plot(plt)
  plt.close('all')
  return image_plot

def get_catastrophic_metric(tasks, metrics):
     forgetting_metrics = Counter()
     count_task = Counter()
     forgetting = {'total': 0, '1_step': 0}
     last_task = tasks[len(tasks) - 1]

     for i,task in enumerate(tasks[:-1]):
        # Calculate forgetting between first trained and last trained task
        current_forgetting = (metrics[tasks[i]][tasks[i]] - metrics[tasks[i]][last_task])
        # Normalize it by it's initial value.
        use_metric = metrics[tasks[i]][tasks[i]]
        if abs(use_metric) == 0:
            # Since use_metric initial learning is at best majority class classifier
            # Hence it's network couldn't learn this task. Make the forgetting highest in this case
            # As Network doesn't have the ability to learn the task now.
            current_forgetting = 1
        else:
            # Calculate the percentage drop This adds the account for % drop and also account if network hasn't learned
            # anything as then value divided is closer to 1 and forgetting increases exponentially and results in highest forgetting.
            # Since we couldn't learn anything.
            current_forgetting = current_forgetting/abs(use_metric)
        # Catastrophic forgetting is always betwen 0 and 1.
        # If out metric is negative then We improved the performance, hence set the forgetting to zero.
        # If network has forgetting greater than 1, Which is when performance is closer to majority class classifier.
        # Then set it to maximum forgetting i.e 1.
        if current_forgetting < 0:
            current_forgetting = 0
        elif current_forgetting > 1:
            current_forgetting = 1
        number_training_steps = (len(tasks) - i - 1)
        forgetting_metrics["1_step"] += (current_forgetting / number_training_steps)
        forgetting_metrics["total"] += current_forgetting
        forgetting_metrics[tasks[i]] = current_forgetting
     
     # Calculate total forgetting of all the
     length_tasks = len(tasks) - 1
     if length_tasks > 1:
         # This is subtracted by 1 as last task never sees any forgetting.
         forgetting_metrics['total'] = forgetting_metrics['total'] 
         forgetting_metrics['1_step'] = (forgetting_metrics['1_step']/(len(tasks) - 1))

     return forgetting_metrics

def torch_remove_neg(matrix):
    mask = matrix < 0
    matrix = matrix.masked_fill(mask, 0)
    return matrix

def get_embedder(type_, vocab, e_dim, rq_grad=False):
    if type_ == 'elmo':
        opt_file = "data/elmo_2x1024_128_2048cnn_1xhighway_options.json"
        wt_file = "data/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5"
        elmo_embedder = ElmoTokenEmbedder(opt_file, wt_file, requires_grad=rq_grad)
        word_embeddings = BasicTextFieldEmbedder({"tokens": elmo_embedder})
        return word_embeddings
    if type_ == 'glove':
        wt_file = "data/glove.6B.300d.txt"
        glove_embedder = Embedding(400000, 300, pretrained_file=wt_file, trainable=rq_grad)
        word_embeddings = BasicTextFieldEmbedder({"tokens": glove_embedder})
        return word_embeddings
    elif type_ == 'bert':
        bert_embedder = PretrainedBertEmbedder(pretrained_model="bert-base-uncased",
                                           top_layer_only=True,requires_grad=rq_grad)
        word_embeddings = BasicTextFieldEmbedder({"tokens": bert_embedder},
                                                                allow_unmatched_keys = True)
        return word_embeddings
    else:
        token_embeddings = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
                                     embedding_dim=e_dim)
        word_embeddings = BasicTextFieldEmbedder({"tokens": token_embeddings})
        return word_embeddings


