import torch

# AllenNLP Imports
from allennlp.modules.seq2vec_encoders import Seq2VecEncoder, PytorchSeq2VecWrapper

## Our Model Imports
from models.cnn_encoder import CnnEncoder
from models.encoder_IDA import EncoderRNN
from models.hashedIDA import HashedMemoryRNN
from models.mlp import MLP
from models.transformer_encoder import TransformerRepresentation
from models.classifier import MainClassifier
from models.other_classifier import Seq2SeqClassifier, MajorityClassifier
from models.mean_classifier import MeanClassifier
import models.utils as utils

def get_model(vocab, word_embeddings, word_embedding_dim, args):
    if args.cnn:
        experiment="cnn_"
        experiment += args.pooling
        ngrams_f=(args.ngram_filter,)
        strides=(args.stride,)
        encoder = CnnEncoder(embedding_dim=word_embedding_dim,
                             num_layers=args.layers,
		             ngram_filter_sizes=ngrams_f,
                             strides=strides,
                             num_filters=args.h_dim,
                             pooling=args.pooling)
    elif args.lstm:
        experiment = "lstm"
        encoder = PytorchSeq2VecWrapper(torch.nn.LSTM(word_embedding_dim, args.h_dim,
                                     num_layers=args.layers,
                                     dropout=args.dropout,
                                     bidirectional=args.bidirectional,
                                     batch_first=True))
    elif args.transformer:
        experiment = "transformer"
        encoder = TransformerRepresentation(word_embedding_dim, # Embedding Dimension
                      8, # Number of heads to use in embeddings.
                      args.h_dim, # Number of hidden units
                      args.layers, # Number of Layers
                      dropout=args.dropout,
                      use_memory=args.use_memory,
                      mem_size=args.mem_size,
                      mem_context_size=args.mem_context_size,
                      use_binary=args.use_binary)
    elif args.IDA:
        experiment = "ida"
        encoder = EncoderRNN(word_embedding_dim, args.h_dim,
                      inv_temp=args.inv_temp,
                      mem_size=args.mem_size,
                      num_layers=args.layers,
                      dropout=args.dropout,
                      bidirectional=args.bidirectional,
                      batch_first=True)
    elif args.hashed:
        experiment = "memory_embeddings"
        memory_embeddings = utils.get_embedder("glove", vocab, word_embedding_dim, rq_grad=False)
        encoder = HashedMemoryRNN(word_embedding_dim, args.h_dim,
                      inv_temp=args.inv_temp,
                      mem_size=args.mem_size,
                      num_layers=args.layers,
                      dropout=args.dropout,
                      bidirectional=args.bidirectional,
                      batch_first=True,
		      memmory_embed=memory_embeddings)
    elif args.mlp:
        experiment = "mlp"
        encoder = MLP(word_embedding_dim,
                      args.h_dim,
                      args.layers,
                      use_binary=args.use_binary)
    else:
        raise "Unknown model"

    model = MainClassifier(word_embeddings, encoder,
                           vocab, inv_temp=args.inv_temp,
                           temp_inc=args.temp_inc,
                           task_embed=args.task_embed,
                           args=args,
                           e_dim=word_embedding_dim)
    if args.majority:
        model = MajorityClassifier(vocab)
    elif args.mean_classifier:
        model = MeanClassifier(word_embeddings, encoder, vocab)
    return model, experiment
