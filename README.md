# Does an LSTM forget more than a CNN? An empirical study of catastrophic forgetting in NLP

This is repository contains code for experiment to evaluate catastrophic forgetting in neural networks for ALTA paper.

## Requirements

* allennlp (Installed from https://github.com/gauravaror/allennlp)
* pytorch
* svcca (https://github.com/google/svcca)
* numpy
* pandas
* 
## Installation Step

1. Create a virtual environment
2. Install [allennlp](https://github.com/gauravaror/allennlp) in virtual environment by running `pip install --editable .`
3. Clone [svcca](https://github.com/google/svcca) folder in home directory of this project.
4. Install latest [pytorch](https://pytorch.org/), Last tested with *1.3(stable)*

## Tasks
Currently, We support running four tasks:
1. TREC (code: trec)
2. SST (code: sst)
3. CoLA (code: cola)
4. Subjectivity (code: subjectivity)

Passing option `--task trec` will add the task. Order of --task option will decide the task in which tasks are trained.


## Architectures
We support running tasks on 
1. CNN (--cnn)
2. LSTM (--seq2vec)
3. GRU (--seq2vec --gru)
4. Transformer Encoder from pytorch. (--seq2vec --transformer)
5. Deep Pyramid CNN (--pyramid)


## Embeddings
You can add embeddings using --embeddings option. Currently supports default(trained from scratch), bert, elmo. But you will have to download the 
ELMo embeddings yourself and store it in data folder.
```
elmo_2x4096_512_2048cnn_2xhighway_options.json  
elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5
```

There are few other options feel free to explore or create an issue on github if you get stuck.

