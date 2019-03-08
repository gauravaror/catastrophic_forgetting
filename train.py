from typing import Iterator, List, Dict
import torch
import torch.optim as optim
import numpy as np

from allennlp.data.dataset_readers import DatasetReader
from allennlp.data.dataset_readers.stanford_sentiment_tree_bank import StanfordSentimentTreeBankDatasetReader
from allennlp.data.vocabulary import Vocabulary

from allennlp.modules.text_field_embedders import TextFieldEmbedder, BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2seq_encoders import Seq2SeqEncoder, PytorchSeq2SeqWrapper
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper
from allennlp.models import Model

from allennlp.training.metrics import CategoricalAccuracy
from allennlp.data.iterators import BucketIterator
from allennlp.training.trainer import Trainer

from allennlp.nn.util import get_text_field_mask, sequence_cross_entropy_with_logits
from allennlp.training.util import move_optimizer_to_cuda
from allennlp.common.params import Params
from models.classifier import MainClassifier
from models.trec import TrecDatasetReader

#reader = StanfordSentimentTreeBankDatasetReader()
reader = TrecDatasetReader()

#train_data = reader.read('data/SST/trees/train.txt')
#dev_data = reader.read('data/SST/trees/dev.txt')

train_data = reader.read('data/Subjectivity/train.txt')
dev_data = reader.read('data/Subjectivity/test.txt')

vocab = Vocabulary.from_instances(train_data + dev_data)

vocab.print_statistics()
EMBEDDING_DIM = 6
HIDDEN_DIM = 6


token_embeddings = Embedding(num_embeddings=vocab.get_vocab_size('tokens'),
	  embedding_dim=EMBEDDING_DIM)

word_embeddings = BasicTextFieldEmbedder({"tokens": token_embeddings})

lstm = PytorchSeq2VecWrapper(torch.nn.LSTM(EMBEDDING_DIM, HIDDEN_DIM, batch_first=True))

model = MainClassifier(word_embeddings, lstm, vocab)

model.cuda(0)


optimizer = optim.SGD(model.parameters(), lr=0.1)

move_optimizer_to_cuda(optimizer)

torch.set_num_threads(5)
iterator = BucketIterator(batch_size=2, sorting_keys=[("tokens", "num_tokens")])

iterator.index_with(vocab)

torch.set_num_threads(4)

trainer = Trainer(model=model,
                  optimizer=optimizer,
                  iterator=iterator,
                  train_dataset=train_data,
                  validation_dataset=dev_data,
                  patience=10,
                  num_epochs=1000,
		  cuda_device=0)

trainer.train()
