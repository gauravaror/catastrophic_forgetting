from typing import Iterator, List, Dict
from allennlp.data import Instance
from allennlp.data.fields import TextField, LabelField

from allennlp.data.dataset_readers import DatasetReader
from allennlp.common.file_utils import cached_path
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token
from allennlp.data.vocabulary import Vocabulary
from allennlp.modules.token_embedders.bert_token_embedder import PretrainedBertEmbedder
from allennlp.modules.token_embedders.elmo_token_embedder import ElmoTokenEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.data.token_indexers.elmo_indexer import ELMoTokenCharactersIndexer
from allennlp.data.token_indexers import PretrainedBertIndexer


class BaseReader(DatasetReader):
    """
    DatasetReader for Trec Question Classification, one sentence per line, like

         LOC:city Tell me what city the Kentucky Horse Park is near ?
    """
    def __init__(self, token_indexers: Dict[str, TokenIndexer] = None,
                 embeddings = 'default', lazy=False, spl=True) -> None:
        super().__init__(lazy=lazy)
        self.embeddings = embeddings
        self.get_token_indexer(token_indexers)
        self._token_indexers = self.token_indexers
        self.spl = spl

    def tokenize(self, s: str):
        if self.embeddings == 'bert':
            return self.token_indexers["bert"].wordpiece_tokenizer(s)[:128 - 2]
        else:
            tokens = s.split()
            if self.spl:
                tokens = ['<CLS>'] + tokens
            return tokens

    def get_token_indexer(self, token_indexers):
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        # the token indexer is responsible for mapping tokens to integers
        if self.embeddings == 'elmo':
            self.token_indexers = {"tokens" : ELMoTokenCharactersIndexer()}
        elif self.embeddings == 'bert':
            self.ber_embedder = PretrainedBertIndexer(pretrained_model="bert-base-uncased",
                                                       max_pieces=128,
                                                       do_lowercase=True,)
            self.token_indexers = {"bert": self.ber_embedder}
